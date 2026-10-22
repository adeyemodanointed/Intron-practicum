"""Fine-tune AfroXLMR as ONE multilingual, multi-label domain classifier (GPU: Kaggle/Colab T4 or better).

    python train_afroxlmr_domain.py --data-dir data --out-dir afroxlmr_out                         # full run
    python train_afroxlmr_domain.py --data-dir data --out-dir smoke --max-train 300 --max-eval 150 --epochs 1   # quick check

Same data, labels, thresholds and scoring as train_domain_classifier.py (TF-IDF baseline):
  labels   agriculture, finance, health, law, telecoms (one sigmoid output each); `other` = no domain passes its threshold;
           at most two domains per text
  training binary cross-entropy per domain; sample weights gold 1.0 / weak 0.5, every language with the same total weight;
           at most --cap-per-cell rows per (language, label) so large weak cells do not dominate (human rows always kept)
  dev      after every epoch: tune per-domain thresholds (shared + per-language), keep the epoch with the best
           mean macro F1 over languages
  test     human-labelled test sets only
Outputs in --out-dir: model/ (weights + tokenizer + thresholds.json), report.md, metrics.csv, test_predictions.csv, train_log.json
"""
import argparse, collections, csv, json, math, os, random, time

import numpy as np
import torch

from domain_model import DOMAINS, LABELS, MAX_LABELS, OTHER
from train_domain_classifier import confusion, score, tune, weights

MODEL = "Davlan/afro-xlmr-base"


def load(data_dir, split):
    return [json.loads(l) for l in open(os.path.join(data_dir, f"domain_{split}.jsonl"), encoding="utf-8")]


def decide(P, langs, thresholds, max_labels=MAX_LABELS):
    out = []
    for row, lang in zip(P, langs):
        th = dict(thresholds["all"]); th.update(thresholds.get(lang, {}))
        picked = [DOMAINS[k] for k in np.argsort(-row) if row[k] >= th[DOMAINS[k]]][:max_labels]
        out.append(picked or [OTHER])
    return out


def cap_rows(rows, cap, seed):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["language"], r["labels"][0])].append(r)
    out = []
    for k, lst in sorted(by.items()):
        gold = [r for r in lst if r["label_source"].startswith("gold")]
        rest = [r for r in lst if not r["label_source"].startswith("gold")]
        rng.shuffle(rest)
        out += gold + rest[: max(0, cap - len(gold))]
    rng.shuffle(out)
    return out


def load_model_and_tokenizer(name):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(DOMAINS), problem_type="multi_label_classification",
        id2label=dict(enumerate(DOMAINS)), label2id={d: i for i, d in enumerate(DOMAINS)})
    return model, tok


def encode(tok, rows, max_len):
    return tok([r["text"] for r in rows], truncation=True, max_length=max_len)["input_ids"]


def batches(ids, batch, shuffle, seed):
    """Length-grouped batches (less padding): shuffle, sort inside pools of 50 batches, shuffle the batches."""
    order = list(range(len(ids)))
    if shuffle:
        random.Random(seed).shuffle(order)
        pools = [sorted(order[i:i + batch * 50], key=lambda j: len(ids[j])) for i in range(0, len(order), batch * 50)]
        bs = [p[i:i + batch] for p in pools for i in range(0, len(p), batch)]
        random.Random(seed + 1).shuffle(bs)
        return bs
    order.sort(key=lambda j: len(ids[j]))
    return [order[i:i + batch] for i in range(0, len(order), batch)]


def collate(ids, idx, pad_id, device):
    L = max(len(ids[j]) for j in idx)
    x = torch.full((len(idx), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(idx), L), dtype=torch.long)
    for i, j in enumerate(idx):
        x[i, : len(ids[j])] = torch.tensor(ids[j]); m[i, : len(ids[j])] = 1
    return x.to(device), m.to(device)


@torch.no_grad()
def predict_proba(model, ids, pad_id, device, batch=64):
    model.eval()
    P = np.zeros((len(ids), len(DOMAINS)), dtype=np.float32)
    for idx in batches(ids, batch, False, 0):
        x, m = collate(ids, idx, pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=x, attention_mask=m).logits
        P[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return P


def eval_split(P, rows, thresholds):
    preds = decide(P, [r["language"] for r in rows], thresholds)
    res = {}
    for l in sorted({r["language"] for r in rows}) + ["ALL"]:
        idx = [i for i, r in enumerate(rows) if (l == "ALL" and r["label_source"].startswith("gold")) or r["language"] == l]
        rs, ps = [rows[i] for i in idx], [preds[i] for i in idx]
        res[l] = (*score(rs, ps), confusion(rs, ps))
    return preds, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--out-dir", default="afroxlmr_out")
    ap.add_argument("--model", default=MODEL); ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--cap-per-cell", type=int, default=4000); ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-train", type=int, default=0, help="quick check: use only this many training rows")
    ap.add_argument("--max-eval", type=int, default=0, help="quick check: use only this many dev and test rows")
    ap.add_argument("--eval-every", type=float, default=1.0, help="check dev every this fraction of an epoch (0.5 = twice per epoch)")
    ap.add_argument("--no-test", action="store_true", help="tuning sweep: score dev only, do not save the model or touch the test set")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    train = cap_rows(load(a.data_dir, "train"), a.cap_per_cell, a.seed)
    dev, test = load(a.data_dir, "dev"), load(a.data_dir, "test")
    if a.max_train: train = train[: a.max_train]
    if a.max_eval:
        rng = random.Random(a.seed); rng.shuffle(dev); rng.shuffle(test); dev, test = dev[: a.max_eval], test[: a.max_eval]
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    print("train rows", len(train), dict(collections.Counter((r["language"], r["labels"][0]) for r in train)))
    print("dev", len(dev), "test", len(test), flush=True)

    model, tok = load_model_and_tokenizer(a.model)
    model.to(device)
    pad_id = tok.pad_token_id
    tr_ids, dv_ids, te_ids = encode(tok, train, a.max_len), encode(tok, dev, a.max_len), encode(tok, test, a.max_len)
    Y = torch.tensor([[d in r["labels"] for d in DOMAINS] for r in train], dtype=torch.float32)
    W = torch.tensor(weights(train), dtype=torch.float32)
    pos = (Y * W[:, None]).sum(0); neg = ((1 - Y) * W[:, None]).sum(0)
    pos_weight = torch.sqrt(neg / pos.clamp(min=1e-6)).clamp(1.0, 20.0).to(device)
    print("pos_weight", dict(zip(DOMAINS, [round(float(x), 2) for x in pos_weight])), flush=True)

    from transformers import get_linear_schedule_with_warmup
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.batch)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    log, best, step, t0 = [], (-1.0, None), 0, time.time()
    best_dir = os.path.join(a.out_dir, "model")

    def check_dev(where):
        nonlocal best
        Pd = predict_proba(model, dv_ids, pad_id, device)
        th = tune(Pd, dev)
        _, res = eval_split(Pd, dev, th)
        langs = [l for l in res if l != "ALL"]
        sel = float(np.mean([res[l][1]["macro_f1"] for l in langs]))
        log.append({"at": where, "dev_mean_macro_f1": round(sel, 4), "dev": {l: res[l][1] for l in res}, "thresholds": th, "minutes": round((time.time() - t0) / 60, 1)})
        print(f"{where} dev mean macro F1 {sel:.3f}", {l: round(float(res[l][1]["macro_f1"]), 3) for l in res}, flush=True)
        model.train()
        if sel > best[0]:
            best = (sel, where)
            np.save(os.path.join(a.out_dir, "dev_probs.npy"), Pd)
            if not a.no_test:
                model.save_pretrained(best_dir); tok.save_pretrained(best_dir)
                json.dump(th, open(os.path.join(best_dir, "thresholds.json"), "w"), indent=1)
            print(f"  best so far ({where})", flush=True)

    for ep in range(1, a.epochs + 1):
        model.train(); run = 0.0
        bl = batches(tr_ids, a.batch, True, a.seed + ep)
        marks = {max(1, round(len(bl) * f)) for f in np.arange(a.eval_every, 1.0 + 1e-9, a.eval_every)} | {len(bl)}
        for bi, idx in enumerate(bl, 1):
            x, m = collate(tr_ids, idx, pad_id, device)
            y, w = Y[idx].to(device), W[idx].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=x, attention_mask=m).logits
            loss = (bce(logits.float(), y).mean(1) * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            run = 0.98 * run + 0.02 * loss.item() if bi > 1 else loss.item()
            if bi % 200 == 0 or bi == len(bl):
                el = time.time() - t0
                print(f"epoch {ep} batch {bi}/{len(bl)} loss {run:.4f} elapsed {el / 60:.1f} min, about {el / step * (steps - step) / 60:.0f} min left", flush=True)
            if bi in marks:
                check_dev(f"epoch {ep - 1 + bi / len(bl):.2f}")
    json.dump(log, open(os.path.join(a.out_dir, "train_log.json"), "w"), indent=1)
    json.dump({"best_dev_mean_macro_f1": round(best[0], 4), "best_at": best[1], "args": vars(a)}, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    if a.no_test:
        print("sweep run: best dev mean macro F1", round(best[0], 4), "at", best[1]); return

    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    th = json.load(open(os.path.join(best_dir, "thresholds.json")))
    P = predict_proba(model, te_ids, pad_id, device)
    np.save(os.path.join(a.out_dir, "test_probs.npy"), P)
    preds, res = eval_split(P, test, th)
    write_outputs(a, test, P, preds, res, th, best, train, dev)
    for l, (_, s, _) in res.items():
        print("TEST", l, {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in s.items()})


def write_outputs(a, test, P, preds, res, th, best, train, dev):
    with open(os.path.join(a.out_dir, "test_predictions.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "language", "source", "human_labels", "predicted_labels", "correct_primary", "exact"] + [f"p_{d}" for d in DOMAINS] + ["text"])
        for r, p, row in zip(test, preds, P):
            w.writerow([r["id"], r["language"], r["source"], "|".join(r["labels"]), "|".join(p), p[0] in r["labels"], set(p) == set(r["labels"])]
                       + [round(float(x), 3) for x in row] + [r["text"]])
    with open(os.path.join(a.out_dir, "metrics.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setup", "language", "label", "precision", "recall", "f1", "support", "predicted"])
        for l, (per, s, _) in res.items():
            for o in per:
                w.writerow(["afroxlmr", l, o["label"], o["precision"], o["recall"], o["f1"], o["support"], o["predicted"]])
            w.writerow(["afroxlmr", l, "SUMMARY", "", "", s["macro_f1"], s["n"], ""])
    L = [f"# AfroXLMR domain classifier: test results", "",
         f"Model `{a.model}`, best checkpoint at {best[1]} of {a.epochs} epochs (dev mean macro F1 {best[0]:.3f}); max length {a.max_len}, batch {a.batch}, lr {a.lr}.",
         f"Training rows {len(train):,} (at most {a.cap_per_cell} per language and label), dev {len(dev):,}, test {len(test):,} (human labels).", "",
         "Thresholds (shared, then per-language overrides):", ""]
    L += [f"- {k}: " + ", ".join(f"{d} {t}" for d, t in v.items()) for k, v in th.items()]
    L += ["", "| Language | Test n | Top label correct | Exact label set | Macro F1 | Labels with test data | Human multi-domain | Predicted multi-domain |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for l, (_, s, _) in res.items():
        L.append(f"| {l} | {s['n']} | {s['primary_acc']:.1%} | {s['exact_match']:.1%} | {s['macro_f1']:.3f} | {s['labels_scored']} of 6 | {s['gold_multi']} | {s['pred_multi']} |")
    for l, (per, s, conf) in res.items():
        L += ["", f"## {l}", "", "| Label | Precision | Recall | F1 | Test support | Predicted |", "|---|---:|---:|---:|---:|---:|"]
        L += [f"| {o['label']} | {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} | {o['support']} | {o['predicted']} |" for o in per]
        L += ["", "| human \\ predicted | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
        L += [f"| {g} | " + " | ".join(str(conf[(g, p)]) for p in LABELS) + " |" for g in LABELS]
    open(os.path.join(a.out_dir, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

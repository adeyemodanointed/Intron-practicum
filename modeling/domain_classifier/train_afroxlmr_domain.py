"""Fine-tune AfroXLMR as ONE multilingual, multi-label domain classifier (GPU: Kaggle/Colab T4 or better).

Two stages, scored by 5-fold cross-validation over domain_eval.jsonl from build_domain_data.py (every human-labelled
text, plus the AI-written Yoruba dev/test rows), the same folds as train_domain_classifier.py --cv:
  stage 1  python train_afroxlmr_domain.py --stage 1 --data-dir data_v3 --out-dir s1 --epochs 2
           trains on domain_train.jsonl only (weak/silver rows and gold training rows) for a fixed number of epochs and
           writes probabilities for every eval row. Eval labels are not used for training or for choosing a checkpoint.
  stage 2  python train_afroxlmr_domain.py --stage 2 --init-from s1/model --fold k --data-dir data_v3 --out-dir s2/fold{k}
           continues on the eval rows of the three training folds (human labels get --gold-share of each language's
           weight) plus a replay sample of domain_train.jsonl (--replay-per-cell rows per language and label, so the
           model keeps what stage 1 taught); dev fold k+1 picks the checkpoint. Writes probabilities for every eval row;
           only test fold k is out-of-fold. --fold -1 = the final model (dev fold 0, trains on folds 1-4).
  scoring  cv_collect.py (thresholds tuned on each dev fold, test fold labelled), then evaluate_domain.py. Stage 1 on
           its own: pass its directory to cv_collect.py (the one model then serves every fold).

Heads (--head)
  multi6    six sigmoid outputs (agriculture, finance, health, law, telecoms, other): several domains can be on at once,
            and `other` is a real output instead of "no domain passed its threshold" (default)
  sigmoid5  five domain outputs, `other` = no domain passes (the v2 model, kept for the ablation)
Loss: binary cross-entropy per output with pos_weight; sample weights gold 1.0 / teacher 0.8 / weak 0.5, every
language with the same total weight. --mix-rate: that share of single-label training rows is replaced by two passages
in the same language with different labels joined into one text (half the length budget each), labelled with both
domains (other + domain -> just the domain), so the model learns to switch on two domains at once.
Large models: --model Davlan/afro-xlmr-large --batch 8 --grad-accum 2 --grad-checkpointing
Outputs in --out-dir: model/ (weights + tokenizer), eval_probs.npy (row order of domain_eval.jsonl), train_log.json,
summary.json. Quick check: --max-train 300 --max-eval 200 --epochs 1.
"""
import argparse, collections, json, math, os, random, time

import numpy as np
import torch

from domain_model import DOMAINS, LABELS, OTHER
from train_domain_classifier import boost, fold_split, mean_macro_f1, tune, weights

MODEL = "Davlan/afro-xlmr-base"


def load(data_dir, split):
    return [json.loads(l) for l in open(os.path.join(data_dir, f"domain_{split}.jsonl"), encoding="utf-8")]


def outputs(head):
    return LABELS if head == "multi6" else DOMAINS


def cap_rows(rows, cap, seed, keep_gold=True):
    """At most `cap` rows per (language, first label); gold rows always kept unless keep_gold=False."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["language"], r["labels"][0])].append(r)
    out = []
    for k, lst in sorted(by.items()):
        gold = [r for r in lst if keep_gold and r["label_source"].startswith("gold")]
        rest = [r for r in lst if not (keep_gold and r["label_source"].startswith("gold"))]
        rng.shuffle(rest)
        out += gold + rest[: max(0, cap - len(gold))]
    rng.shuffle(out)
    return out


def load_model_and_tokenizer(name, head):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    labs = outputs(head)
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(num_labels=len(labs), problem_type="multi_label_classification",
              id2label=dict(enumerate(labs)), label2id={d: i for i, d in enumerate(labs)})
    try:   # float32 master weights: some checkpoints (e.g. afro-xlmr-large) are stored in fp16, which GradScaler cannot train
        model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32, **kw)
    except TypeError:   # older transformers
        model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32, **kw)
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


def collate(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s); m[i, : len(s)] = 1
    return x.to(device), m.to(device)


@torch.no_grad()
def predict_proba(model, ids, pad_id, device, batch=64):
    model.eval()
    P = np.zeros((len(ids), model.config.num_labels), dtype=np.float32)
    for idx in batches(ids, batch, False, 0):
        x, m = collate([ids[j] for j in idx], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=x, attention_mask=m).logits
        P[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return P


class Mixer:
    """Replaces a share of single-label rows by two same-language passages with different labels joined together."""
    def __init__(self, rows, ids, Y, W, labs, rate, max_len, seed):
        self.rows, self.ids, self.Y, self.W, self.labs, self.rate, self.max_len = rows, ids, Y, W, labs, rate, max_len
        self.rng = random.Random(seed)
        self.by_lang = collections.defaultdict(list)
        for i, r in enumerate(rows):
            if len(r["labels"]) == 1:
                self.by_lang[r["language"]].append(i)

    def batch(self, idx):
        seqs, ys, ws = [], [], []
        for i in idx:
            j = self.partner(i) if self.rate and len(self.rows[i]["labels"]) == 1 and self.rng.random() < self.rate else None
            if j is None:
                seqs.append(self.ids[i]); ys.append(self.Y[i]); ws.append(self.W[i])
                continue
            a, b = self.ids[i][1:-1], self.ids[j][1:-1]
            if self.rng.random() < 0.5:
                a, b = b, a
            h = (self.max_len - 2) // 2
            seqs.append([self.ids[i][0]] + a[:h] + b[:h] + [self.ids[i][-1]])
            labs = (set(self.rows[i]["labels"]) | set(self.rows[j]["labels"])) - {OTHER}
            ys.append(torch.tensor([l in labs for l in self.labs], dtype=torch.float32))
            ws.append(torch.minimum(self.W[i], self.W[j]))
        return seqs, torch.stack(ys), torch.stack(ws)

    def partner(self, i):
        lst = self.by_lang[self.rows[i]["language"]]
        for _ in range(8):
            j = lst[self.rng.randrange(len(lst))]
            if self.rows[j]["labels"] != self.rows[i]["labels"]:
                return j
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_v3"); ap.add_argument("--out-dir", default="afroxlmr_out")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--init-from", help="stage 2: the stage-1 model directory")
    ap.add_argument("--fold", type=int, default=0, help="stage 2: test fold (dev = fold + 1); -1 = final model")
    ap.add_argument("--model", default=MODEL); ap.add_argument("--head", choices=("multi6", "sigmoid5"), default="multi6")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1); ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--cap-per-cell", type=int, default=15000, help="stage 1: at most this many rows per language and label")
    ap.add_argument("--replay-per-cell", type=int, default=300, help="stage 2: domain_train.jsonl rows per language and label")
    ap.add_argument("--gold-share", type=float, default=0.6, help="stage 2: share of each language's weight for human labels")
    ap.add_argument("--mix-rate", type=float, default=0.1, help="share of single-label rows replaced by two-label mixes")
    ap.add_argument("--eval-every", type=float, default=0.5, help="stage 2: check the dev fold every this fraction of an epoch")
    ap.add_argument("--patience", type=int, default=4, help="stage 2: stop after this many dev checks without a gain")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-train", type=int, default=0, help="quick check: use only this many training rows")
    ap.add_argument("--max-eval", type=int, default=0, help="quick check: use only this many eval rows")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    labs = outputs(a.head)

    pool, ev = load(a.data_dir, "train"), load(a.data_dir, "eval")
    if a.max_eval:
        ev = random.Random(a.seed).sample(ev, min(a.max_eval, len(ev)))
    tr, dv, te = fold_split(ev, a.fold if a.stage == 2 else 0)
    if a.stage == 1:
        train, dev = cap_rows(pool, a.cap_per_cell, a.seed), []
        W = weights(train)
    else:
        replay = cap_rows(pool, a.replay_per_cell, a.seed, keep_gold=False)
        train = [ev[i] for i in tr] + replay
        mask = np.array([ev[i]["label_source"].startswith("gold") for i in tr] + [False] * len(replay))
        W = boost(weights(train), train, mask, a.gold_share)
        dev = [ev[i] for i in dv]
        order = list(range(len(train))); random.Random(a.seed).shuffle(order)
        train, W = [train[i] for i in order], W[order]
    if a.max_train:
        train, W = train[: a.max_train], W[: a.max_train]
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    print(f"stage {a.stage}, fold {a.fold}, head {a.head}: train rows {len(train)}", dict(collections.Counter((r["language"], r["labels"][0]) for r in train)))
    print("dev rows", len(dev), "eval rows", len(ev), flush=True)

    model, tok = load_model_and_tokenizer(a.init_from or a.model, a.head)
    if a.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    pad_id = tok.pad_token_id
    tr_ids, ev_ids = encode(tok, train, a.max_len), encode(tok, ev, a.max_len)
    dv_ids = [ev_ids[i] for i in dv]
    Y = torch.tensor([[l in r["labels"] for l in labs] for r in train], dtype=torch.float32)
    W = torch.tensor(W, dtype=torch.float32)
    pos = (Y * W[:, None]).sum(0); neg = ((1 - Y) * W[:, None]).sum(0)
    pos_weight = torch.sqrt(neg / pos.clamp(min=1e-6)).clamp(1.0, 20.0).to(device)
    print("pos_weight", dict(zip(labs, [round(float(x), 2) for x in pos_weight])), flush=True)
    mixer = Mixer(train, tr_ids, Y, W, labs, a.mix_rate, a.max_len, a.seed)

    from transformers import get_linear_schedule_with_warmup
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    n_batches = math.ceil(len(train) / a.batch)
    steps = math.ceil(a.epochs * n_batches / a.grad_accum)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    log, best, step, t0, stale = [], (-1.0, None), 0, time.time(), 0
    best_dir = os.path.join(a.out_dir, "model")

    def check_dev(where):
        nonlocal best, stale
        Pd = predict_proba(model, dv_ids, pad_id, device)
        th = tune(Pd, dev)
        sel = mean_macro_f1(Pd, dev, th)
        log.append({"at": where, "dev_mean_macro_f1": round(sel, 4), "minutes": round((time.time() - t0) / 60, 1)})
        print(f"{where} dev mean macro F1 {sel:.3f}", flush=True)
        model.train()
        if sel > best[0]:
            best, stale = (sel, where), 0
            model.save_pretrained(best_dir); tok.save_pretrained(best_dir)
            print(f"  best so far ({where})", flush=True)
        else:
            stale += 1

    total = math.ceil(a.epochs * n_batches)
    done, ep = 0, 0
    while done < total and not (a.stage == 2 and stale >= a.patience):
        ep += 1
        model.train()
        bl = batches(tr_ids, a.batch, True, a.seed + ep)[: total - done]
        marks = {max(1, round(n_batches * f)) for f in np.arange(a.eval_every, 1.0 + 1e-9, a.eval_every)} | {len(bl)}
        run = 0.0
        for bi, idx in enumerate(bl, 1):
            seqs, y, w = mixer.batch(idx)
            x, m = collate(seqs, pad_id, device)
            y, w = y.to(device), w.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=x, attention_mask=m).logits
            loss = (bce(logits.float(), y).mean(1) * w).sum() / w.sum()
            scaler.scale(loss / a.grad_accum).backward()
            if bi % a.grad_accum == 0 or bi == len(bl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            run = 0.98 * run + 0.02 * loss.item() if bi > 1 else loss.item()
            if bi % 200 == 0 or bi == len(bl):
                el = time.time() - t0
                print(f"epoch {ep} batch {bi}/{len(bl)} loss {run:.4f} elapsed {el / 60:.1f} min, about {el / (done + bi) * (total - done - bi) / 60:.0f} min left", flush=True)
            if a.stage == 2 and bi in marks:
                check_dev(f"epoch {ep - 1 + bi / n_batches:.2f}")
                if stale >= a.patience:
                    print("no dev gain for", a.patience, "checks: stopping", flush=True)
                    break
        done += bi
    if a.stage == 1:
        model.save_pretrained(best_dir); tok.save_pretrained(best_dir)
        best = (float("nan"), f"epoch {done / n_batches:.2f} (fixed)")

    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    P = predict_proba(model, ev_ids, pad_id, device)
    np.save(os.path.join(a.out_dir, "eval_probs.npy"), P)
    json.dump([r["id"] for r in ev], open(os.path.join(a.out_dir, "eval_ids.json"), "w"))
    check = {}
    if a.stage == 1:   # sanity number only (fold 0, thresholds from fold 1); not used to choose anything
        _, dv0, te0 = fold_split(ev, 0)
        th = tune(P[dv0], [ev[i] for i in dv0])
        check = {"fold0_mean_macro_f1": round(mean_macro_f1(P[te0], [ev[i] for i in te0], th), 4)}
    elif te:
        th = tune(P[dv], dev)
        check = {"test_fold_mean_macro_f1": round(mean_macro_f1(P[te], [ev[i] for i in te], th), 4)}
    json.dump(log, open(os.path.join(a.out_dir, "train_log.json"), "w"), indent=1)
    json.dump({"best_dev_mean_macro_f1": None if math.isnan(best[0]) else round(best[0], 4), "best_at": best[1], **check,
               "train_rows": len(train), "minutes": round((time.time() - t0) / 60, 1), "args": vars(a)},
              open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print("done", best, check, flush=True)


if __name__ == "__main__":
    main()

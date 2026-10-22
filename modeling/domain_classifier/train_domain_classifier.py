"""Train and evaluate the multilingual multi-label domain classifier.

    python train_domain_classifier.py              # one model for every language
    python train_domain_classifier.py --compare    # also train one model per language on the same data, for comparison
    python train_domain_classifier.py --augment    # add per-language feature copies (tried; no clear gain, see README)

Input : data/domain_{train,dev,test}.jsonl (from build_domain_data.py)
Output: model/domain_classifier.joblib, model/thresholds.json
        results/metrics.csv, results/test_predictions.csv, results/report.md

Sample weights: gold 1.0, teacher 0.8, weak 0.5; then rescaled so every language has the same total weight
(Kinyarwanda has 2-5x more training rows than the others).
Thresholds: one per domain, tuned on dev for that domain's F1: a shared set (each language counts equally),
then a per-language override wherever that language's dev set has at least MIN_DEV_POS examples of the domain.
Scores are on the human test sets; ALL pools human-labelled rows only (the Yoruba test is synthetic and reported on its own). A text counts as correct for "primary" if the top prediction is one of
its human labels, and for "exact" if the predicted set equals the human set.
"""
import argparse, collections, csv, json, os, time

import joblib
import numpy as np

from domain_model import DOMAINS, LABELS, OTHER, DomainClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
KIND_W = {"gold": 1.0, "teacher": 0.8, "weak": 0.5}
MIN_DEV_POS = 5


def load(split):
    return [json.loads(l) for l in open(os.path.join(HERE, "data", f"domain_{split}.jsonl"), encoding="utf-8")]


def weights(rows):
    w = np.array([KIND_W[r["label_source"].split(":")[0]] for r in rows])
    langs = [r["language"] for r in rows]
    tot = collections.Counter()
    for l, x in zip(langs, w):
        tot[l] += x
    target = w.sum() / len(tot)
    return w * np.array([target / tot[l] for l in langs])


def tune(P, rows):
    """Shared thresholds (languages weighted equally) plus per-language overrides where dev has enough examples."""
    th = {"all": {d: 0.5 for d in DOMAINS} | tune_one(P, rows, min_pos=1)}
    for l in sorted({r["language"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["language"] == l]
        own = tune_one(P[idx], [rows[i] for i in idx], min_pos=MIN_DEV_POS)
        if own:
            th[l] = own
    return th


def tune_one(P, rows, min_pos):
    """Per-domain threshold maximising that domain's F1, with each language weighted equally."""
    langs = [r["language"] for r in rows]
    n = collections.Counter(langs)
    lw = np.array([1.0 / n[l] for l in langs])
    th = {}
    for k, d in enumerate(DOMAINS):
        y = np.array([d in r["labels"] for r in rows])
        if y.sum() < min_pos:
            continue
        best = (-1, 0.5)
        for t in np.arange(0.05, 0.96, 0.025):
            p = P[:, k] >= t
            tp, fp, fn = (lw * (p & y)).sum(), (lw * (p & ~y)).sum(), (lw * (~p & y)).sum()
            f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f > best[0] + 1e-9:
                best = (f, round(float(t), 3))
        th[d] = best[1]
    return th


def score(rows, preds):
    """Per-label precision/recall/F1 (multi-label sets) plus primary and exact-match accuracy."""
    out = []
    for lab in LABELS:
        tp = sum(lab in r["labels"] and lab in p for r, p in zip(rows, preds))
        fp = sum(lab not in r["labels"] and lab in p for r, p in zip(rows, preds))
        fn = sum(lab in r["labels"] and lab not in p for r, p in zip(rows, preds))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append({"label": lab, "precision": round(pr, 3), "recall": round(rc, 3),
                    "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0, "support": tp + fn, "predicted": tp + fp})
    sup = [o for o in out if o["support"]]
    summ = {"n": len(rows),
            "primary_acc": round(np.mean([p[0] in r["labels"] for r, p in zip(rows, preds)]), 3),
            "exact_match": round(np.mean([set(p) == set(r["labels"]) for r, p in zip(rows, preds)]), 3),
            "macro_f1": round(np.mean([o["f1"] for o in sup]), 3),
            "gold_multi": sum(len(r["labels"]) > 1 for r in rows), "pred_multi": sum(len(p) > 1 for p in preds),
            "labels_scored": len(sup)}
    return out, summ


def confusion(rows, preds):
    m = collections.Counter((r["labels"][0] if p[0] not in r["labels"] else p[0], p[0]) for r, p in zip(rows, preds))
    return {(g, p): m[(g, p)] for g in LABELS for p in LABELS}


def fit_eval(train, dev, test, tag, augment=False):
    t0 = time.time()
    m = DomainClassifier(augment=augment).fit([r["text"] for r in train], [r["language"] for r in train], [r["labels"] for r in train], weights(train))
    m.thresholds = tune(m.proba([r["text"] for r in dev], [r["language"] for r in dev]), dev)
    P = m.proba([r["text"] for r in test], [r["language"] for r in test])
    preds = m.decide(P, [r["language"] for r in test])
    print(f"[{tag}] trained on {len(train)} rows in {time.time() - t0:.0f}s; thresholds {m.thresholds}", flush=True)
    return m, P, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="also train a separate model per language")
    ap.add_argument("--augment", action="store_true", help="shared + per-language feature copies")
    a = ap.parse_args()
    train, dev, test = load("train"), load("dev"), load("test")
    langs = sorted({r["language"] for r in test})
    os.makedirs(os.path.join(HERE, "model"), exist_ok=True); os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    model, P, preds = fit_eval(train, dev, test, "one model", a.augment)
    joblib.dump(model, os.path.join(HERE, "model", "domain_classifier.joblib"), compress=3)
    json.dump(model.thresholds, open(os.path.join(HERE, "model", "thresholds.json"), "w"), indent=1)

    results = {}   # (setup, language) -> (per-label, summary, confusion)
    for l in langs + ["ALL"]:
        idx = [i for i, r in enumerate(test) if (l == "ALL" and r["label_source"].startswith("gold")) or r["language"] == l]
        rs, ps = [test[i] for i in idx], [preds[i] for i in idx]
        results[("one model", l)] = (*score(rs, ps), confusion(rs, ps))

    if a.compare:
        for l in langs:
            tr = [r for r in train if r["language"] == l]; dv = [r for r in dev if r["language"] == l]; te = [r for r in test if r["language"] == l]
            _, _, ps = fit_eval(tr, dv, te, f"{l} only")
            results[("per-language model", l)] = (*score(te, ps), confusion(te, ps))

    with open(os.path.join(HERE, "results", "test_predictions.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "language", "source", "human_labels", "predicted_labels", "correct_primary", "exact"] + [f"p_{d}" for d in DOMAINS] + ["text"])
        for r, p, row in zip(test, preds, P):
            w.writerow([r["id"], r["language"], r["source"], "|".join(r["labels"]), "|".join(p), p[0] in r["labels"], set(p) == set(r["labels"])]
                       + [round(float(x), 3) for x in row] + [r["text"]])
    with open(os.path.join(HERE, "results", "metrics.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setup", "language", "label", "precision", "recall", "f1", "support", "predicted"])
        for (setup, l), (per, summ, _) in results.items():
            for o in per:
                w.writerow([setup, l, o["label"], o["precision"], o["recall"], o["f1"], o["support"], o["predicted"]])
            w.writerow([setup, l, "SUMMARY", "", "", summ["macro_f1"], summ["n"], ""])
    report(results, langs, model.thresholds, train, dev, test)
    for (setup, l), (_, summ, _) in results.items():
        print(setup, l, summ)


def report(results, langs, th, train, dev, test):
    L = ["# Multilingual domain classifier: results", "",
         "One model for every language. Labels: agriculture, finance, health, law, telecoms; a text gets one or two domains, "
         "or `other` when no domain reaches its threshold.", "",
         "Model: word 1-2 gram + character 3-5 gram TF-IDF (lower-cased, diacritics removed, language tag token) and one "
         "class-balanced logistic regression per domain. Test sets are human-labelled except Yoruba (synthetic, unreviewed); `ALL` pools the human-labelled rows only.", "",
         "## Data", "", "| Language | Train rows | Dev rows (threshold tuning) | Test rows |", "|---|---:|---:|---:|"]
    for l in langs:
        L.append(f"| {l} | {sum(r['language'] == l for r in train):,} | {sum(r['language'] == l for r in dev):,} | {sum(r['language'] == l for r in test):,} |")
    L += ["", "Thresholds (shared, then per-language overrides tuned on that language's dev set):", ""]
    L += [f"- {k}: " + ", ".join(f"{d} {t}" for d, t in v.items()) for k, v in th.items()]
    L += ["",
          "## Summary", "", "| Setup | Language | Test n | Primary label correct | Exact label set | Macro F1 | Labels with test data | Human multi-domain | Predicted multi-domain |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (setup, l), (_, s, _) in results.items():
        L.append(f"| {setup} | {l} | {s['n']} | {s['primary_acc']:.1%} | {s['exact_match']:.1%} | {s['macro_f1']:.3f} | {s['labels_scored']} of 6 | {s['gold_multi']} | {s['pred_multi']} |")
    for l in langs + ["ALL"]:
        per, s, conf = results[("one model", l)]
        L += ["", f"## One model, {l}", "", "| Label | Precision | Recall | F1 | Test support | Predicted |", "|---|---:|---:|---:|---:|---:|"]
        L += [f"| {o['label']} | {o['precision']:.3f} | {o['recall']:.3f} | {o['f1']:.3f} | {o['support']} | {o['predicted']} |" for o in per]
        L += ["", "Top prediction vs human label (rows: human, columns: predicted):", "",
              "| human \\ predicted | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
        for g in LABELS:
            L.append(f"| {g} | " + " | ".join(str(conf[(g, p)]) for p in LABELS) + " |")
    open(os.path.join(HERE, "results", "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

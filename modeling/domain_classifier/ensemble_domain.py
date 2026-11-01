"""Combine several AfroXLMR runs (different seeds) and, optionally, the TF-IDF model into one classifier
(v2 dev/test protocol; the cross-validated v3 runs use cv_collect.py).

    python ensemble_domain.py --runs runs/seed13 runs/seed14 runs/seed15 --tfidf model/domain_classifier.joblib --out results/ensemble

Each run directory needs dev_probs.npy and test_probs.npy (written by train_afroxlmr_domain.py, same data files).
Steps: average the AfroXLMR runs; mix with TF-IDF as (1 - w) * AfroXLMR + w * TF-IDF, choosing w on dev
from 0, 0.25, 0.5, 0.75, 1; tune per-domain thresholds on dev; score the test set once.
Writes report.md, metrics.csv, test_predictions.csv and thresholds.json (with the chosen weight) to --out.
"""
import argparse, csv, json, os

import numpy as np

from domain_model import DOMAINS, decide_labels
from train_domain_classifier import confusion, score, tune

HERE = os.path.dirname(os.path.abspath(__file__))


def load(data_dir, split):
    return [json.loads(l) for l in open(os.path.join(data_dir, f"domain_{split}.jsonl"), encoding="utf-8")]


def eval_split(P, rows, thresholds):
    preds = decide_labels(P, [r["language"] for r in rows], thresholds)
    res = {}
    for l in sorted({r["language"] for r in rows}) + ["ALL"]:
        idx = [i for i, r in enumerate(rows) if (l == "ALL" and r["label_source"].startswith("gold")) or r["language"] == l]
        rs, ps = [rows[i] for i in idx], [preds[i] for i in idx]
        res[l] = (*score(rs, ps), confusion(rs, ps))
    return preds, res


def mean_dev_f1(P, rows):
    th = tune(P, rows)
    _, res = eval_split(P, rows, th)
    return float(np.mean([res[l][1]["macro_f1"] for l in res if l != "ALL"])), th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tfidf", help="TF-IDF joblib model to mix in")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ensemble"))
    a = ap.parse_args()
    dev, test = load(a.data_dir, "dev"), load(a.data_dir, "test")
    Pd = np.mean([np.load(os.path.join(r, "dev_probs.npy")) for r in a.runs], axis=0)
    Pt = np.mean([np.load(os.path.join(r, "test_probs.npy")) for r in a.runs], axis=0)
    assert Pd.shape[0] == len(dev) and Pt.shape[0] == len(test), "probabilities do not match these data files"
    w, tried = 0.0, {}
    if a.tfidf:
        import joblib
        m = joblib.load(a.tfidf)
        Td = m.proba([r["text"] for r in dev], [r["language"] for r in dev])
        Tt = m.proba([r["text"] for r in test], [r["language"] for r in test])
        for cand in (0.0, 0.25, 0.5, 0.75, 1.0):
            tried[cand] = round(mean_dev_f1((1 - cand) * Pd + cand * Td, dev)[0], 4)
        w = max(tried, key=tried.get)
        Pd, Pt = (1 - w) * Pd + w * Td, (1 - w) * Pt + w * Tt
    sel, th = mean_dev_f1(Pd, dev)
    preds, res = eval_split(Pt, test, th)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"tfidf_weight": w, "runs": a.runs, "thresholds": th}, open(os.path.join(a.out, "thresholds.json"), "w"), indent=1)
    with open(os.path.join(a.out, "test_predictions.csv"), "w", encoding="utf-8-sig", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["id", "language", "source", "human_labels", "predicted_labels", "correct_primary", "exact"] + [f"p_{d}" for d in DOMAINS] + ["text"])
        for r, p, row in zip(test, preds, Pt):
            cw.writerow([r["id"], r["language"], r["source"], "|".join(r["labels"]), "|".join(p), p[0] in r["labels"], set(p) == set(r["labels"])]
                        + [round(float(x), 3) for x in row] + [r["text"]])
    with open(os.path.join(a.out, "metrics.csv"), "w", encoding="utf-8", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["language", "label", "precision", "recall", "f1", "support", "predicted"])
        for l, (per, s, _) in res.items():
            for o in per:
                cw.writerow([l, o["label"], o["precision"], o["recall"], o["f1"], o["support"], o["predicted"]])
            cw.writerow([l, "SUMMARY", "", "", s["macro_f1"], s["n"], ""])
    L = ["# Ensemble domain classifier: test results", "",
         f"AfroXLMR runs averaged: {len(a.runs)}. TF-IDF weight {w} (dev mean macro F1 by weight: {tried or 'not used'}). "
         f"Dev mean macro F1 {sel:.3f}.", "",
         "| Language | Test n | Top label correct | Exact label set | Macro F1 |", "|---|---:|---:|---:|---:|"]
    for l, (_, s, _) in res.items():
        L.append(f"| {l} | {s['n']} | {s['primary_acc']:.1%} | {s['exact_match']:.1%} | {s['macro_f1']:.3f} |")
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

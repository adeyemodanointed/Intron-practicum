"""Turn cross-validation probabilities into one out-of-fold prediction file (every eval row predicted by a model that
never trained on it), optionally averaging runs and mixing in a second model.

    python cv_collect.py --eval data_v3/domain_eval.jsonl --run results/cv/afroxlmr_large --out results/cv_scored/afroxlmr_large
    python cv_collect.py --run runs/s2_seed13 runs/s2_seed14 --mix results/cv/tfidf6 --out results/cv_scored/ensemble

A run directory holds fold{k}/eval_probs.npy (one model per fold), or a single eval_probs.npy used for every fold
(a model trained without eval rows, e.g. stage 1). Probabilities follow the row order of --eval.
For each fold k: average the --run probabilities; if --mix is given, combine (1 - w) * runs + w * mix with w chosen on
the dev fold (k + 1) from 0, 0.25, 0.5, 0.75, 1; tune thresholds on the dev fold; label the test fold k.
Five-output and six-output models can be combined: the `other` column is dropped unless every model has it.
Writes test_predictions.csv (for evaluate_domain.py) and cv_thresholds.json to --out.
"""
import argparse, csv, json, os

import numpy as np

from domain_model import DOMAINS, MAX_LABELS, OTHER, decide_labels
from train_domain_classifier import N_FOLDS, fold_split, mean_macro_f1, tune

HERE = os.path.dirname(os.path.abspath(__file__))


def probs(run, k):
    for p in (os.path.join(run, f"fold{k}", "eval_probs.npy"), os.path.join(run, "eval_probs.npy")):
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(f"{run}: no fold{k}/eval_probs.npy or eval_probs.npy")


def package(a, rows):
    parts = [np.load(os.path.join(r, "eval_probs.npy")) for r in a.run] + ([np.load(os.path.join(a.mix, "eval_probs.npy"))] if a.mix else [])
    width = min(p.shape[1] for p in parts)
    P = np.mean([p[:, :width] for p in parts[: len(a.run)]], axis=0)
    dv = [i for i, r in enumerate(rows) if r["fold"] == 0]
    dev = [rows[i] for i in dv]
    w, tried = 0.0, {}
    if a.mix:
        M = parts[-1][:, :width]
        for cand in (0.0, 0.25, 0.5, 0.75, 1.0):
            Q = (1 - cand) * P + cand * M
            tried[cand] = round(mean_macro_f1(Q[dv], dev, tune(Q[dv], dev)), 4)
        w = max(tried, key=tried.get)
        P = (1 - w) * P + w * M
    th = tune(P[dv], dev)
    os.makedirs(a.package, exist_ok=True)
    rel = lambda p: os.path.relpath(p, a.package).replace(os.sep, "/")
    cfg = {"models": [rel(os.path.join(r, "model")) for r in a.run],
           "tfidf": rel(os.path.join(a.mix, "domain_classifier.joblib")) if a.mix else None,
           "tfidf_weight": w, "mix_tried": tried, "max_labels": a.max_labels, "thresholds": th,
           "dev_fold0_mean_macro_f1": round(mean_macro_f1(P[dv], dev, th), 4)}
    json.dump(cfg, open(os.path.join(a.package, "predictor.json"), "w"), indent=1)
    print("wrote", os.path.join(a.package, "predictor.json"), "| TF-IDF weight", w, "| dev fold 0 mean macro F1", cfg["dev_fold0_mean_macro_f1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(HERE, "data_v3", "domain_eval.jsonl"))
    ap.add_argument("--run", nargs="+", required=True, help="run directories, averaged")
    ap.add_argument("--mix", help="second model mixed in with a weight chosen on each dev fold")
    ap.add_argument("--max-labels", type=int, default=MAX_LABELS)
    ap.add_argument("--out", help="out-of-fold predictions directory")
    ap.add_argument("--package", help="instead: final models (--fold -1 runs and the TF-IDF final/ directory); tune the mix "
                                      "weight and thresholds on their dev fold 0 and write predictor.json here")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.eval, encoding="utf-8")]
    if a.package:
        return package(a, rows)
    langs = [r["language"] for r in rows]
    preds, final, log = [None] * len(rows), np.zeros((len(rows), len(DOMAINS) + 1), np.float32), {}
    for k in range(N_FOLDS):
        parts = [probs(r, k) for r in a.run] + ([probs(a.mix, k)] if a.mix else [])
        width = min(p.shape[1] for p in parts)
        parts = [p[:, :width] for p in parts]
        for p in parts:
            assert p.shape[0] == len(rows), "probabilities do not match --eval"
        P = np.mean(parts[: len(a.run)], axis=0)
        _, dv, te = fold_split(rows, k)
        dev = [rows[i] for i in dv]
        w, tried = 0.0, {}
        if a.mix:
            M = parts[-1]
            for cand in (0.0, 0.25, 0.5, 0.75, 1.0):
                Q = (1 - cand) * P + cand * M
                tried[cand] = round(mean_macro_f1(Q[dv], dev, tune(Q[dv], dev)), 4)
            w = max(tried, key=tried.get)
            P = (1 - w) * P + w * M
        th = tune(P[dv], dev)
        for i, p in zip(te, decide_labels(P[te], [langs[i] for i in te], th, a.max_labels)):
            preds[i] = p
        final[te, :width] = P[te]
        log[k] = {"mix_weight": w, "mix_tried": tried, "dev_mean_macro_f1": round(mean_macro_f1(P[dv], dev, th), 4), "thresholds": th}
        print(f"fold {k}: mix weight {w}, dev mean macro F1 {log[k]['dev_mean_macro_f1']}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"runs": a.run, "mix": a.mix, "folds": log}, open(os.path.join(a.out, "cv_thresholds.json"), "w"), indent=1)
    cols = DOMAINS + ([OTHER] if width > len(DOMAINS) else [])
    with open(os.path.join(a.out, "test_predictions.csv"), "w", encoding="utf-8-sig", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["id", "language", "fold", "source", "human_labels", "predicted_labels", "correct_primary", "exact"] + [f"p_{c}" for c in cols] + ["text"])
        for r, p, row in zip(rows, preds, final):
            cw.writerow([r["id"], r["language"], r["fold"], r["source"], "|".join(r["labels"]), "|".join(p), p[0] in r["labels"],
                        set(p) == set(r["labels"])] + [round(float(x), 3) for x in row[: len(cols)]] + [r["text"]])
    print("wrote", os.path.join(a.out, "test_predictions.csv"))


if __name__ == "__main__":
    main()

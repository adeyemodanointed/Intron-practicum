"""Score the final models on the independent team test set (make_team_test.py), counting only texts the model never
trained on (overlap_<version> false).

    python score_teamtest.py       # -> results/eval_teamtest/{predictions/<system>/test_predictions.csv, report.md, ...}

Systems: the TF-IDF and AfroXLMR final models (probabilities from the Kaggle kernel domain-v3-teamtest, which loads
them with the library versions they were trained with). Thresholds are tuned exactly as for the delivered model: on
dev fold 0 of the final model's own eval probabilities (cv_collect.py --package logic).
"""
import csv, json, os, subprocess, sys

import numpy as np

from domain_model import DOMAINS, LABELS, decide_labels
from train_domain_classifier import mean_macro_f1, tune

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.join(HERE, "results", "kaggle_v3")
OUT = os.path.join(HERE, "results", "eval_teamtest")
TEST = os.path.join(HERE, "data_teamtest", "domain_teamtest.jsonl")
EV3 = os.path.join(HERE, "data_v3", "domain_eval.jsonl")
# name: (final eval_probs for thresholds, eval file, team-test probabilities, training data version)
SYSTEMS = {
    "tfidf6_final": (f"{K}/domain-v3-tfidf/cv/tfidf6/final/eval_probs.npy", EV3, f"{K}/domain-v3-teamtest/teamtest/tfidf6_probs.npy", "v3"),
    "base_final_v3m": (f"{K}/domain-v3-final-base/runs/base/final/eval_probs.npy", EV3, f"{K}/domain-v3-teamtest/teamtest/base_probs.npy", "v3m"),
    "large_final_v3m": (f"{K}/domain-v3-final-large/runs/large/final/eval_probs.npy", EV3, f"{K}/domain-v3-teamtest/teamtest/large_probs.npy", "v3m"),
}


def main():
    rows = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    langs = [r["language"] for r in rows]
    preds = {}
    for name, (evp, evf, src, ver) in SYSTEMS.items():
        if not (os.path.exists(evp) and os.path.exists(src)):
            print("skip (missing files):", name); continue
        ev = [json.loads(l) for l in open(evf, encoding="utf-8")]
        Pe = np.load(evp)
        dv = [i for i, r in enumerate(ev) if r["fold"] == 0]
        th = tune(Pe[dv], [ev[i] for i in dv])
        P = np.load(src)[:, : Pe.shape[1]]
        labs = decide_labels(P, langs, th)
        keep = [i for i, r in enumerate(rows) if not r[f"overlap_{ver}"]]
        d = os.path.join(OUT, "predictions", name); os.makedirs(d, exist_ok=True)
        cols = LABELS[: P.shape[1]] if P.shape[1] > len(DOMAINS) else DOMAINS
        with open(os.path.join(d, "test_predictions.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "language", "source", "human_labels", "predicted_labels", "correct_primary", "exact"] + [f"p_{c}" for c in cols] + ["text"])
            for i in keep:
                r, p = rows[i], labs[i]
                w.writerow([r["id"], r["language"], r["source"], "|".join(r["labels"]), "|".join(p), p[0] in r["labels"], set(p) == set(r["labels"])]
                           + [round(float(x), 3) for x in P[i]] + [r["text"]])
        preds[name] = os.path.join(d, "test_predictions.csv")
        print(name, "scored on", len(keep), "texts it never trained on; dev fold 0 mean macro F1", round(mean_macro_f1(Pe[dv], [ev[i] for i in dv], th), 4))
    if preds:
        subprocess.run([sys.executable, "evaluate_domain.py", "--pred", *[f"{n}={p}" for n, p in preds.items()], "--data", TEST,
                        "--only-common", "--baseline", "tfidf6_final" if "tfidf6_final" in preds else list(preds)[0], "--out", OUT],
                       cwd=HERE, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


if __name__ == "__main__":
    main()

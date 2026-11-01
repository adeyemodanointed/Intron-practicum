"""Test post-hoc "other" gates on saved out-of-fold probabilities (no retraining): does an energy-based
out-of-distribution gate beat the current rule (no domain passes, or p(other) beats the best domain by a margin)?

    python posthoc_other_gate.py --eval data_v8/domain_eval.jsonl --run results/kaggle_v3/domain-v8-base/runs/v8/stage2

Gates (each tuned on the dev fold k + 1 for mean per-language macro F1, then applied to test fold k):
  current     the delivered rule (cv_collect.py)
  energy      softmax energy  -T * logsumexp(z / T) over the five domain logits; above tau -> other
  joint       multi-label JointEnergy  -sum log(1 + exp(z_k)); above tau -> other
  p_other     p(other) above tau -> other (instead of the margin)
tau is tuned per language (shared value where the dev fold has too few `other` texts).
"""
import argparse, json, os

import numpy as np

from cv_collect import probs
from domain_model import DOMAINS, OTHER, decide_labels
from train_domain_classifier import MIN_DEV_POS, N_FOLDS, fold_split, score, tune

VALID = "team_manual_validation"


def logits(P):
    p = np.clip(P[:, : len(DOMAINS)], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def gate_score(P, kind, T=1.0):
    if kind == "energy":
        z = logits(P) / T
        m = z.max(1, keepdims=True)
        return -T * (m[:, 0] + np.log(np.exp(z - m).sum(1)))
    if kind == "joint":
        return -np.logaddexp(0, logits(P)).sum(1)
    if kind == "p_other":
        return P[:, len(DOMAINS)]
    raise ValueError(kind)


def apply(base, s, langs, tau):
    return [[OTHER] if s[i] > tau.get(langs[i], tau["all"]) else p for i, p in enumerate(base)]


def lang_f1(rows, preds):
    ls = sorted({r["language"] for r in rows})
    return float(np.mean([score([r for r in rows if r["language"] == l], [p for r, p in zip(rows, preds) if r["language"] == l])[1]["macro_f1"] for l in ls]))


def tune_gate(s, base, rows):
    langs = [r["language"] for r in rows]
    grid = np.unique(np.quantile(s, np.linspace(0.3, 1.0, 71)))
    grid = np.append(grid, np.inf)   # inf = gate off
    tau = {"all": float(max(grid, key=lambda t: lang_f1(rows, apply(base, s, langs, {"all": t}))))}
    for l in sorted(set(langs)):
        ix = [i for i, x in enumerate(langs) if x == l]
        if sum(OTHER in rows[i]["labels"] for i in ix) < MIN_DEV_POS:
            continue
        sub, b, sl = [rows[i] for i in ix], [base[i] for i in ix], s[ix]
        tau[l] = float(max(grid, key=lambda t: lang_f1(sub, apply(b, sl, [l] * len(ix), {"all": t}))))
    return tau


def report(rows, preds, name):
    out = {}
    for grp, keep in [("ALL", lambda r: True), ("ALL (validated)", lambda r: r["label_source"].endswith(VALID)),
                      ("pcm (validated)", lambda r: r["language"] == "pcm" and r["label_source"].endswith(VALID)),
                      ("kin", lambda r: r["language"] == "kin")]:
        ix = [i for i, r in enumerate(rows) if keep(r)]
        sub, pr = [rows[i] for i in ix], [preds[i] for i in ix]
        acc = np.mean([p[0] in r["labels"] for r, p in zip(sub, pr)])
        oth = [p[0] == OTHER for r, p in zip(sub, pr) if r["labels"][0] == OTHER]
        out[grp] = {"n": len(ix), "top_label_acc": round(100 * acc, 1), "other_acc": round(100 * np.mean(oth), 1) if oth else None,
                    "macro_f1": round(lang_f1(sub, pr), 3)}
    print(f"{name:10s}", " | ".join(f"{g}: acc {v['top_label_acc']} other {v['other_acc']} F1 {v['macro_f1']}" for g, v in out.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data_v8/domain_eval.jsonl")
    ap.add_argument("--run", default="results/kaggle_v3/domain-v8-base/runs/v8/stage2")
    ap.add_argument("--out", default="results/eval_v8/other_gate.json")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.eval, encoding="utf-8")]
    langs = [r["language"] for r in rows]
    variants = ["current", "energy_T1", "energy_T2", "joint", "p_other", "energy_T1_no_margin"]
    preds = {v: [None] * len(rows) for v in variants}
    for k in range(N_FOLDS):
        P = probs(a.run, k)
        _, dv, te = fold_split(rows, k)
        dev = [rows[i] for i in dv]
        th = tune(P[dv], dev)
        no_margin = {l: {**t, "other_margin": 9.0} for l, t in th.items()}
        base_dv, base_te = decide_labels(P[dv], [langs[i] for i in dv], th), decide_labels(P[te], [langs[i] for i in te], th)
        nm_dv, nm_te = decide_labels(P[dv], [langs[i] for i in dv], no_margin), decide_labels(P[te], [langs[i] for i in te], no_margin)
        for i, p in zip(te, base_te):
            preds["current"][i] = p
        for v, kind, T, (bd, bt) in [("energy_T1", "energy", 1.0, (base_dv, base_te)), ("energy_T2", "energy", 2.0, (base_dv, base_te)),
                                     ("joint", "joint", 1.0, (base_dv, base_te)), ("p_other", "p_other", 1.0, (nm_dv, nm_te)),
                                     ("energy_T1_no_margin", "energy", 1.0, (nm_dv, nm_te))]:
            tau = tune_gate(gate_score(P[dv], kind, T), bd, dev)
            for i, p in zip(te, apply(bt, gate_score(P[te], kind, T), [langs[i] for i in te], tau)):
                preds[v][i] = p
    res = {v: report(rows, preds[v], v) for v in variants}
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

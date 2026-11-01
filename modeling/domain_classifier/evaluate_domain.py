"""Score one or more prediction files the same way: per-domain accuracy, multi-domain metrics, confidence intervals,
and paired tests between systems.

    python evaluate_domain.py --pred tfidf=results/test_predictions.csv ensemble=results/ensemble/test_predictions.csv \
        --baseline tfidf --out results/eval

Each prediction file is a CSV with columns id, language, human_labels, predicted_labels (labels joined by "|", top
prediction first), as written by train_domain_classifier.py, train_afroxlmr_domain.py, ensemble_domain.py and
llm_baseline_domain.py. --data (default data/domain_test.jsonl) supplies label_source, so ALL pools human-labelled
rows only; rows missing from it count as human-labelled unless their language is in --synthetic.

Metrics per language (rows that are not human-labelled, e.g. AI-written Yoruba, form their own "(synthetic)" group)
and for ALL (human-labelled rows):
  top-label accuracy   the top prediction is one of the human labels
  per-domain accuracy  of the texts a person labelled with that domain, the share the model also labels with it
                       (recall); precision = of the texts the model gives that domain, the share that are right
  balanced accuracy    mean per-domain accuracy over the labels that have test texts (each domain counts equally,
                       so the many Pidgin "other" texts do not dominate)
  macro F1             mean per-label F1 over labels with test texts
  exact set            the predicted set equals the human set
  label-set F1         per-text F1 between predicted and human sets, averaged over texts
  multi-domain         texts with two or more human domains: both found (every human domain predicted), exact set;
                       single-domain texts: false second-label rate (a second label was predicted)
95% intervals: bootstrap over texts (--boot resamples). Paired test against --baseline: bootstrap of the difference
on the same texts (two-sided p) plus an exact McNemar test on top-label correctness.
Writes summary.csv, per_domain.csv, multi_domain.csv, paired.csv and report.md to --out.
"""
import argparse, csv, json, math, os

import numpy as np

from domain_model import LABELS, OTHER

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = {l: i for i, l in enumerate(LABELS)}


def split_labels(s):
    return [x for x in str(s or "").split("|") if x] or [OTHER]


def load_preds(path, sources, synthetic):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        src = sources.get(r["id"])
        gold = src.startswith("gold") if src else r["language"] not in synthetic
        rows.append({"id": r["id"], "language": r["language"], "gold": gold, "team": bool(src and src.endswith("team_manual_validation")),
                     "human": split_labels(r["human_labels"]), "pred": split_labels(r["predicted_labels"])})
    return rows


class Arrays:
    """Binary label matrices for fast bootstrap: H (human), P (predicted), top (index of top prediction)."""
    def __init__(self, rows):
        n = len(rows)
        self.H = np.zeros((n, len(LABELS)), bool); self.P = np.zeros((n, len(LABELS)), bool)
        self.top = np.array([IDX[r["pred"][0]] for r in rows])
        for i, r in enumerate(rows):
            self.H[i, [IDX[x] for x in r["human"]]] = True
            self.P[i, [IDX[x] for x in r["pred"]]] = True
        self.top_ok = self.H[np.arange(n), self.top]
        self.multi = self.H.sum(1) > 1

    def metrics(self, idx=None):
        H, P, ok = (self.H, self.P, self.top_ok) if idx is None else (self.H[idx], self.P[idx], self.top_ok[idx])
        tp = (H & P).sum(0); sup = H.sum(0); pred = P.sum(0)
        has = sup > 0
        rec = np.divide(tp, sup, out=np.zeros(len(LABELS)), where=has)
        prec = np.divide(tp, pred, out=np.zeros(len(LABELS)), where=pred > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(LABELS)), where=(prec + rec) > 0)
        inter = (H & P).sum(1); set_f1 = 2 * inter / (H.sum(1) + P.sum(1))
        return {"top_acc": ok.mean(), "balanced_acc": rec[has].mean(), "macro_f1": f1[has].mean(),
                "exact_set": (H == P).all(1).mean(), "label_set_f1": set_f1.mean()}


def group(r):
    if r.get("team"):
        return f"{r['language']} (validated)"
    return r["language"] if r["gold"] else f"{r['language']} (synthetic)"


def groups(rows):
    out = {g: [i for i, r in enumerate(rows) if group(r) == g] for g in sorted({group(r) for r in rows})}
    out["ALL"] = [i for i, r in enumerate(rows) if r["gold"]]
    if any(r.get("team") for r in rows):
        out["ALL (validated)"] = [i for i, r in enumerate(rows) if r.get("team")]
    return out


def boot_ci(A, idx, B, rng):
    idx = np.asarray(idx)
    draws = [A.metrics(idx[rng.integers(0, len(idx), len(idx))]) for _ in range(B)]
    return {k: (np.percentile([d[k] for d in draws], 2.5), np.percentile([d[k] for d in draws], 97.5)) for k in draws[0]}


def per_domain(rows, idx):
    out = []
    for lab in LABELS:
        sup = [i for i in idx if lab in rows[i]["human"]]
        got = [i for i in idx if lab in rows[i]["pred"]]
        tp = sum(lab in rows[i]["pred"] for i in sup)
        acc = tp / len(sup) if sup else None
        prec = tp / len(got) if got else None
        f1 = 2 * acc * prec / (acc + prec) if acc and prec else (0.0 if sup else None)
        out.append({"label": lab, "n": len(sup), "accuracy": acc, "precision": prec, "f1": f1, "predicted": len(got)})
    return out


def multi_domain(rows, idx):
    multi = [i for i in idx if len(rows[i]["human"]) > 1]
    single = [i for i in idx if len(rows[i]["human"]) == 1]
    both = sum(set(rows[i]["human"]) <= set(rows[i]["pred"]) for i in multi)
    exact = sum(set(rows[i]["human"]) == set(rows[i]["pred"]) for i in multi)
    false2 = sum(len(rows[i]["pred"]) > 1 for i in single)
    return {"multi_n": len(multi), "both_found": both, "multi_exact": exact,
            "single_n": len(single), "false_second": false2, "pred_multi": sum(len(rows[i]["pred"]) > 1 for i in idx)}


def mcnemar_p(a_ok, b_ok):
    """Exact two-sided McNemar test on paired correct/incorrect outcomes."""
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum()); n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(A, Bm, idx, B, rng):
    idx = np.asarray(idx)
    base_a, base_b = A.metrics(idx), Bm.metrics(idx)
    diffs = {k: [] for k in base_a}
    for _ in range(B):
        s = idx[rng.integers(0, len(idx), len(idx))]
        ma, mb = A.metrics(s), Bm.metrics(s)
        for k in diffs:
            diffs[k].append(ma[k] - mb[k])
    out = {}
    for k, d in diffs.items():
        d = np.array(d)
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        out[k] = (base_a[k] - base_b[k], np.percentile(d, 2.5), np.percentile(d, 97.5), min(1.0, p))
    out["mcnemar_p_top_acc"] = mcnemar_p(A.top_ok[idx], Bm.top_ok[idx])
    return out


def pct(x):
    return "" if x is None else f"{x:.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True, help="name=path.csv, one per system")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "domain_test.jsonl"))
    ap.add_argument("--synthetic", nargs="*", default=["yor"], help="languages whose rows are not human-labelled when --data has no entry")
    ap.add_argument("--baseline", help="system name to compare every other system against")
    ap.add_argument("--only-common", action="store_true", help="score only the texts that every system labelled")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "eval"))
    a = ap.parse_args()
    sources = {}
    if a.data and os.path.exists(a.data):
        for l in open(a.data, encoding="utf-8"):
            r = json.loads(l); sources[r["id"]] = r["label_source"]
    systems = {}
    for spec in a.pred:
        name, path = spec.split("=", 1)
        systems[name] = load_preds(path, sources, set(a.synthetic))
    os.makedirs(a.out, exist_ok=True)

    names = list(systems)
    ids0 = [r["id"] for r in systems[names[0]]]
    if a.only_common:   # e.g. LLMs scored on the human-labelled rows only
        common = set.intersection(*({r["id"] for r in rows} for rows in systems.values()))
        ids0 = [i for i in ids0 if i in common]
        print(f"scoring the {len(ids0)} texts every system labelled")
    for n in names:   # same texts in the same order, so paired tests line up
        by = {r["id"]: r for r in systems[n]}
        missing = [i for i in ids0 if i not in by]
        assert not missing, f"{n} lacks {len(missing)} ids of {names[0]} (use --only-common)"
        systems[n] = [by[i] for i in ids0]

    summary, perdom, multi, pair = [], [], [], []
    for n, rows in systems.items():
        A = Arrays(rows); rng = np.random.default_rng(a.seed)
        for g, idx in groups(rows).items():
            if not idx:
                continue
            m = A.metrics(np.asarray(idx)); ci = boot_ci(A, idx, a.boot, rng) if a.boot else {k: (np.nan, np.nan) for k in m}
            summary.append({"system": n, "language": g, "n": len(idx), **{k: m[k] for k in m},
                            **{f"{k}_lo": ci[k][0] for k in m}, **{f"{k}_hi": ci[k][1] for k in m}})
            perdom += [{"system": n, "language": g, **o} for o in per_domain(rows, idx)]
            multi.append({"system": n, "language": g, **multi_domain(rows, idx)})
    if a.baseline:
        Bm = Arrays(systems[a.baseline])
        for n in names:
            if n == a.baseline:
                continue
            A = Arrays(systems[n]); rng = np.random.default_rng(a.seed)
            for g, idx in groups(systems[n]).items():
                if idx:
                    r = paired(A, Bm, idx, a.boot or 1000, rng)
                    pair.append({"system": n, "vs": a.baseline, "language": g, **{f"{k}_diff": v[0] for k, v in r.items() if k != "mcnemar_p_top_acc"},
                                 **{f"{k}_p": v[3] for k, v in r.items() if k != "mcnemar_p_top_acc"}, "mcnemar_p_top_acc": r["mcnemar_p_top_acc"]})

    def write(name, rows):
        if not rows:
            return
        with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    write("summary.csv", summary); write("per_domain.csv", perdom); write("multi_domain.csv", multi); write("paired.csv", pair)

    L = ["# Domain classifier evaluation", "",
         "Top-label accuracy: the top prediction is one of the human labels. Balanced accuracy: the mean of the per-domain "
         "accuracies, so every domain counts equally. Brackets: 95% bootstrap interval. `ALL` = human-labelled texts only.", "",
         "## Summary", "", "| System | Language | n | Top-label accuracy | Balanced accuracy | Macro F1 | Exact set | Label-set F1 |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        ci = lambda k: f" [{s[k + '_lo']:.1%}–{s[k + '_hi']:.1%}]" if not np.isnan(s[k + "_lo"]) else ""
        L.append(f"| {s['system']} | {s['language']} | {s['n']} | {s['top_acc']:.1%}{ci('top_acc')} | {s['balanced_acc']:.1%}{ci('balanced_acc')} | "
                 f"{s['macro_f1']:.3f} | {s['exact_set']:.1%} | {s['label_set_f1']:.3f} |")
    L += ["", "## Accuracy per domain", "",
          "Accuracy = of the texts a person gave this domain, the share the model also gave it. Precision = of the texts the "
          "model gave this domain, the share that are right. Blank = no texts of that domain in this test set.", ""]
    for n in names:
        L += [f"### {n}", "", "| Language | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
        for g in groups(systems[n]):
            cells = {o["label"]: o for o in perdom if o["system"] == n and o["language"] == g}
            if not cells:
                continue
            L.append(f"| {g} | " + " | ".join(f"{pct(cells[l]['accuracy'])} (n={cells[l]['n']})" if cells[l]["n"] else "" for l in LABELS) + " |")
        L.append("")
    L += ["## Multi-domain texts", "", "| System | Language | Two-domain texts | Both domains found | Exact set | Single-domain texts | Given a wrong second label |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for m in multi:
        if m["multi_n"] or m["pred_multi"]:
            L.append(f"| {m['system']} | {m['language']} | {m['multi_n']} | {m['both_found']} | {m['multi_exact']} | {m['single_n']} | "
                     f"{m['false_second']} ({m['false_second'] / max(1, m['single_n']):.1%}) |")
    if pair:
        L += ["", f"## Paired comparison against {a.baseline}", "",
              "Difference = system minus baseline on the same texts; p from the paired bootstrap (two-sided); McNemar = exact test on top-label correctness.", "",
              "| System | Language | Top-label accuracy diff (p) | Balanced accuracy diff (p) | Macro F1 diff (p) | McNemar p |", "|---|---|---:|---:|---:|---:|"]
        for p in pair:
            L.append(f"| {p['system']} | {p['language']} | {p['top_acc_diff']:+.1%} ({p['top_acc_p']:.3f}) | {p['balanced_acc_diff']:+.1%} ({p['balanced_acc_p']:.3f}) | "
                     f"{p['macro_f1_diff']:+.3f} ({p['macro_f1_p']:.3f}) | {p['mcnemar_p_top_acc']:.3f} |")
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

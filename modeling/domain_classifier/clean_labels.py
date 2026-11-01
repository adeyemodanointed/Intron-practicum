"""Find likely wrong labels among the weak (source/keyword) training rows with confident learning (Northcutt et al.,
2021), using out-of-fold TF-IDF probabilities, so no GPU is needed.

    python clean_labels.py --data-dir data_v8 --out results/label_cleaning_v8                       # find issues
    python clean_labels.py --data-dir data_v8 --out results/label_cleaning_v8 --write data_v9_clean  # also write cleaned data

1. Split domain_train.jsonl into 5 folds (stratified by language and first label). For each fold, fit the six-output
   TF-IDF model (domain_model.DomainClassifier, same features and sample weights as the baseline) on the other four
   and predict it. Eval rows (the human test texts) are never used.
2. Per language and label j, the self-confidence threshold t_j is the mean p_j over that language's rows labelled j.
3. A row is a label issue when none of its labels reaches its threshold but another label does; the suggested label
   is the highest-scoring of those. Only weak and teacher rows are dropped; gold rows are only reported.
Writes oof_probs.npy (row order of domain_train.jsonl), issues.csv (every flagged row with its text) and report.md.
"""
import argparse, collections, csv, json, os, shutil, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from domain_model import LABELS, DomainClassifier
from train_domain_classifier import load, weights

N_FOLDS, SEED, MIN_ROWS = 5, 13, 5
DROPPABLE = ("weak", "teacher")


def folds(rows):
    """Stable stratified folds: within each (language, first label) group, a seeded shuffle dealt round-robin."""
    f = np.zeros(len(rows), int)
    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["language"], r["labels"][0])].append(i)
    rng = np.random.default_rng(SEED)
    for key in sorted(groups):
        ix = np.array(groups[key])[rng.permutation(len(groups[key]))]
        f[ix] = np.arange(len(ix)) % N_FOLDS
    return f


def fit_fold(data_dir, out, k):
    path = os.path.join(out, f"oof_fold{k}.npy")
    if os.path.exists(path):
        return k, "already done"
    rows = load("train", data_dir)
    f = folds(rows)
    tr, te = np.where(f != k)[0], np.where(f == k)[0]
    t0 = time.time()
    sub = [rows[i] for i in tr]
    m = DomainClassifier(other_output=True).fit([r["text"] for r in sub], [r["language"] for r in sub], [r["labels"] for r in sub], weights(sub))
    np.save(path, m.proba([rows[i]["text"] for i in te], [rows[i]["language"] for i in te]).astype(np.float32))
    return k, f"{len(tr)} train / {len(te)} predicted rows, {time.time() - t0:.0f}s"


def find_issues(rows, P):
    langs = np.array([r["language"] for r in rows])
    Y = np.array([[lab in r["labels"] for lab in LABELS] for r in rows])
    t_all = np.array([P[Y[:, j], j].mean() for j in range(len(LABELS))])
    th = {}
    for l in sorted(set(langs)):
        m = langs == l
        th[l] = np.array([P[m & Y[:, j], j].mean() if (m & Y[:, j]).sum() >= MIN_ROWS else t_all[j] for j in range(len(LABELS))])
    T = np.stack([th[l] for l in langs])
    passes = P >= T
    issue = ~(passes & Y).any(1) & (passes & ~Y).any(1)
    sugg = np.where(passes & ~Y, P, -1).argmax(1)
    return issue, sugg, {l: dict(zip(LABELS, np.round(t, 3).tolist())) for l, t in th.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_v8")
    ap.add_argument("--out", default="results/label_cleaning_v8")
    ap.add_argument("--jobs", type=int, default=3, help="folds fitted at once (each needs a few GB of memory)")
    ap.add_argument("--write", help="also write a copy of --data-dir without the flagged weak/teacher rows here")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for k, msg in ex.map(fit_fold, [a.data_dir] * N_FOLDS, [a.out] * N_FOLDS, range(N_FOLDS)):
            print(f"fold {k}: {msg}", flush=True)
    rows = load("train", a.data_dir)
    f = folds(rows)
    P = np.zeros((len(rows), len(LABELS)), np.float32)
    for k in range(N_FOLDS):
        P[f == k] = np.load(os.path.join(a.out, f"oof_fold{k}.npy"))
    np.save(os.path.join(a.out, "oof_probs.npy"), P)
    issue, sugg, th = find_issues(rows, P)
    kind = np.array([r["label_source"].split(":")[0] for r in rows])
    drop = issue & np.isin(kind, DROPPABLE)

    with open(os.path.join(a.out, "issues.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "language", "source", "label_source", "given", "suggested", "p_given", "p_suggested", "dropped", "text"])
        for i in np.where(issue)[0]:
            r = rows[i]
            w.writerow([r["id"], r["language"], r.get("source", ""), r["label_source"], "|".join(r["labels"]), LABELS[sugg[i]],
                        round(max(float(P[i, LABELS.index(x)]) for x in r["labels"]), 3), round(float(P[i, sugg[i]]), 3), bool(drop[i]), r["text"]])

    lines = ["# Label cleaning (confident learning on out-of-fold TF-IDF probabilities)", "",
             f"Data: `{a.data_dir}/domain_train.jsonl`, {len(rows):,} rows. Flagged: {int(issue.sum()):,}; dropped (weak or teacher labels only): {int(drop.sum()):,}.", "",
             "## Dropped rows by language and label", "", "| Language | Given label | Rows (weak/teacher) | Dropped | Share | Most often suggested |", "|---|---|---:|---:|---:|---|"]
    for l in sorted(set(r["language"] for r in rows)):
        for lab in LABELS:
            ix = [i for i, r in enumerate(rows) if r["language"] == l and r["labels"][0] == lab and kind[i] in DROPPABLE]
            if not ix:
                continue
            d = [i for i in ix if drop[i]]
            top = ", ".join(f"{LABELS[s]} {n}" for s, n in collections.Counter(sugg[d]).most_common(3))
            lines.append(f"| {l} | {lab} | {len(ix):,} | {len(d):,} | {100 * len(d) / len(ix):.1f}% | {top} |")
    lines += ["", "## Dropped rows by source (sources with at least 100 weak/teacher rows)", "", "| Language | Source | Rows | Dropped | Share |", "|---|---|---:|---:|---:|"]
    src = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if kind[i] in DROPPABLE:
            src[(r["language"], r.get("source", "?"))].append(i)
    for (l, s), ix in sorted(src.items(), key=lambda kv: -np.mean(drop[kv[1]])):
        if len(ix) >= 100:
            lines.append(f"| {l} | {s} | {len(ix):,} | {int(drop[ix].sum()):,} | {100 * drop[ix].mean():.1f}% |")
    lines += ["", "## Self-confidence thresholds", "", "```", json.dumps(th, indent=1), "```"]
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))

    if a.write:
        os.makedirs(a.write, exist_ok=True)
        for fn in os.listdir(a.data_dir):
            if fn != "domain_train.jsonl" and os.path.isfile(os.path.join(a.data_dir, fn)):
                shutil.copy(os.path.join(a.data_dir, fn), a.write)
        with open(os.path.join(a.write, "domain_train.jsonl"), "w", encoding="utf-8") as fh:
            for r, d in zip(rows, drop):
                if not d:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        json.dump({"from": a.data_dir, "dropped": int(drop.sum()), "kept": int((~drop).sum()), "method": "clean_labels.py"},
                  open(os.path.join(a.write, "cleaning_log.json"), "w"), indent=1)
        print("wrote", a.write, "without", int(drop.sum()), "rows")


if __name__ == "__main__":
    main()

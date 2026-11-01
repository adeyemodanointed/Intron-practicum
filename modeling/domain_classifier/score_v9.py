"""Score the v9 experiments (FineWeb-2 off-topic texts, label cleaning) against v8 on the same out-of-fold texts.

    python score_v9.py      # -> results/eval_v9/{report.md, summary.csv, per_domain.csv, paired.csv, by_source.md}

Systems that exist are scored; the eval texts and folds of data_v8 and data_v9 are identical.
  v8 (earlier):  base_v8, tfidf6_v8 (gold share chosen per fold from 0.15/0.3/0.5), ensemble_v8
  TF-IDF ablation (gold share 0.5, run locally): tfidf6_{v8clean,v9,v9clean} in results/cv_ablation/
  AfroXLMR-base v9 (FineWeb-2 + cleaning, Kaggle domain-v9-base): base_v9clean, and base_v9clean + tfidf6_v9clean
by_source.md splits the validated texts into FineWeb-2-sourced ones (v9 now trains on other FineWeb-2 pages) and the rest.
"""
import collections, csv, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "data_v8", "domain_eval.jsonl")
OUT = os.path.join(HERE, "results", "eval_v9")
P3 = os.path.join(HERE, "results", "eval_v3", "predictions")
OLD = {"base_v8": f"{P3}/base_v8_final/test_predictions.csv", "tfidf6_v8": f"{P3}/tfidf6_v8/test_predictions.csv",
       "ensemble_v8": f"{P3}/ensemble_v8/test_predictions.csv"}
AB = os.path.join(HERE, "results", "cv_ablation")
B9 = os.path.join(HERE, "results", "kaggle_v3", "domain-v9-base", "runs", "v9", "stage2")
NEW = {"tfidf6_v8clean": ([f"{AB}/tfidf6_v8clean"], None), "tfidf6_v9": ([f"{AB}/tfidf6_v9"], None),
       "tfidf6_v9clean": ([f"{AB}/tfidf6_v9clean"], None), "base_v9clean": ([B9], None),
       "ensemble_v9clean": ([B9], f"{AB}/tfidf6_v9clean")}


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=HERE, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def complete(d):
    return all(os.path.exists(os.path.join(d, f"fold{k}", "eval_probs.npy")) for k in range(5))


def by_source(preds):
    rows = {r["id"]: r for r in map(json.loads, open(EV, encoding="utf-8"))}
    lines = ["# Validated texts by source", "", "Top-label accuracy on the team-validated texts. v9 trains on FineWeb-2 pages "
             "(none of them containing a test text), so FineWeb-2-sourced test texts are reported apart.", "",
             "| System | FineWeb-2 texts | Other validated texts | Validated `other` (FineWeb-2) | Validated `other` (rest) |", "|---|---:|---:|---:|---:|"]
    for name, path in preds.items():
        acc = collections.defaultdict(list)
        for p in csv.DictReader(open(path, encoding="utf-8-sig")):
            r = rows.get(p["id"])
            if not r or not r["label_source"].endswith("team_manual_validation"):
                continue
            g = "fw" if "FineWeb" in r["source"] else "rest"
            ok = p["predicted_labels"].split("|")[0] in r["labels"]
            acc[g].append(ok)
            if r["labels"][0] == "other":
                acc[g + "_other"].append(ok)
        f = lambda k: f"{100 * sum(acc[k]) / len(acc[k]):.1f}% (n={len(acc[k])})" if acc[k] else "-"
        lines.append(f"| {name} | {f('fw')} | {f('rest')} | {f('fw_other')} | {f('rest_other')} |")
    open(os.path.join(OUT, "by_source.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    os.makedirs(OUT, exist_ok=True)
    preds = {n: p for n, p in OLD.items() if os.path.exists(p)}
    for name, (runs, mix) in NEW.items():
        if not all(complete(r) for r in runs) or (mix and not complete(mix)):
            print("skip (not finished):", name); continue
        d = os.path.join(OUT, "predictions", name)
        run("cv_collect.py", "--eval", EV, "--run", *runs, *(["--mix", mix] if mix else []), "--out", d)
        preds[name] = os.path.join(d, "test_predictions.csv")
    run("evaluate_domain.py", "--pred", *[f"{n}={p}" for n, p in preds.items()], "--data", EV, "--only-common",
        "--baseline", "base_v8", "--out", OUT)
    by_source(preds)


if __name__ == "__main__":
    main()

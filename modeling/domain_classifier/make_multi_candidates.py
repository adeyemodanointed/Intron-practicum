"""Multi-domain data: find likely two-domain training texts, turn LLM labels into multi-domain training rows, and make
review sheets whose human answers become the multi-domain test set.

    python make_multi_candidates.py candidates --data-dir data_v3 --per-lang 750      # -> data_v3/multi_candidates.jsonl
    (label them: llm_baseline_domain.py --input data_v3/multi_candidates.jsonl --all-rows --shots 6 --out llm/multi)
    python make_multi_candidates.py from-llm --data-dir data_v3 --llm llm/multi --review-per-lang 60
        -> data_v3/multi_silver.jsonl (LLM two-domain rows, label_source weak:llm_multi, for build_domain_data.py --relabel)
        -> review/multi_domain_review_<lang>.csv (sheets for native speakers; these rows are kept out of multi_silver)
    python make_multi_candidates.py from-review --data-dir data_v3 --review review/*.csv
        -> data_v3/domain_multi_test.jsonl (human-confirmed labels; score with evaluate_domain.py)

Candidates: the TF-IDF model (model/domain_classifier.joblib) scores every training row; a row is a candidate when its
best domain has probability >= --top-min and its second-best >= --second-min (raw probabilities: some tuned thresholds
sit at the edge of the grid). The --per-lang rows per language with the highest second-best probability are kept
(texts of 15+ words).
"""
import argparse, collections, csv, glob, json, os, random

import joblib
import numpy as np

from domain_model import DOMAINS, OTHER, norm_labels

HERE = os.path.dirname(os.path.abspath(__file__))


def candidates(a):
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, "domain_train.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if len(r["text"].split()) >= 15]
    m = joblib.load(a.model)
    P = m.proba([r["text"] for r in rows], [r["language"] for r in rows])[:, : len(DOMAINS)]
    by = collections.defaultdict(list)
    for r, p in zip(rows, P):
        o = np.argsort(-p)
        if p[o[0]] >= a.top_min and p[o[1]] >= a.second_min:
            by[r["language"]].append((p[o[1]], r, p))
    out = []
    for l, lst in sorted(by.items()):
        lst.sort(key=lambda x: -x[0])
        for s2, r, p in lst[: a.per_lang]:
            out.append({**{k: r[k] for k in ("id", "language", "text", "labels", "label_source", "source")},
                        "candidate_score": round(float(s2), 3), **{f"p_{d}": round(float(x), 3) for d, x in zip(DOMAINS, p)}})
        print(l, "candidates", len(lst), "kept", min(len(lst), a.per_lang))
    path = os.path.join(a.data_dir, "multi_candidates.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(out), "->", path)


def from_llm(a):
    cand = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(a.data_dir, "multi_candidates.jsonl"), encoding="utf-8")}
    llm = {r["id"]: r["predicted_labels"].split("|") for r in csv.DictReader(open(os.path.join(a.llm, "test_predictions.csv"), encoding="utf-8-sig"))}
    rng = random.Random(13)
    review, silver = collections.defaultdict(list), []
    by = collections.defaultdict(list)
    for i, labs in llm.items():
        by[cand[i]["language"]].append(i)
    for l, ids in sorted(by.items()):
        rng.shuffle(ids)
        multi = [i for i in ids if len(llm[i]) > 1]
        single = [i for i in ids if len(llm[i]) == 1]
        k = a.review_per_lang
        held = set(multi[: k * 2 // 3] + single[: k - min(len(multi), k * 2 // 3)])   # mostly two-domain, some one-domain
        review[l] = [cand[i] | {"llm_domains": "|".join(llm[i])} for i in ids if i in held]
        silver += [{"language": l, "text": cand[i]["text"], "labels": llm[i], "label_source": "weak:llm_multi", "source": cand[i]["source"]}
                   for i in multi if i not in held]
        print(l, "LLM two-domain", len(multi), "of", len(ids), "| review sheet", len(review[l]), "| silver", sum(s["language"] == l for s in silver))
    with open(os.path.join(a.data_dir, "multi_silver.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in silver:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.makedirs(a.review_dir, exist_ok=True)
    for l, lst in review.items():
        with open(os.path.join(a.review_dir, f"multi_domain_review_{l}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "language", "text", "suggested_domains", "your_domains", "notes"])
            for r in lst:
                w.writerow([r["id"], l, r["text"], r["llm_domains"], "", ""])
    print("review sheets ->", a.review_dir, "| fill `your_domains` with one or two of", ", ".join(DOMAINS), "or other")


def from_review(a):
    out = []
    for p in sorted(sum((glob.glob(g) for g in a.review), [])):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            if r["your_domains"].strip():
                out.append({"id": f"multi_{r['id']}", "language": r["language"], "text": r["text"], "labels": norm_labels(r["your_domains"]),
                            "split": "test", "source": os.path.basename(p), "label_source": "gold:human_multi_review", "fold": -1})
    path = os.path.join(a.data_dir, "domain_multi_test.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(out), "rows,", sum(len(r["labels"]) > 1 for r in out), "with two domains ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("candidates", "from-llm", "from-review"))
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data_v3"))
    ap.add_argument("--model", default=os.path.join(HERE, "model", "domain_classifier.joblib"))
    ap.add_argument("--per-lang", type=int, default=750)
    ap.add_argument("--top-min", type=float, default=0.6); ap.add_argument("--second-min", type=float, default=0.4)
    ap.add_argument("--llm"); ap.add_argument("--review-per-lang", type=int, default=60)
    ap.add_argument("--review-dir", default=os.path.join(HERE, "review")); ap.add_argument("--review", nargs="*", default=[])
    a = ap.parse_args()
    {"candidates": candidates, "from-llm": from_llm, "from-review": from_review}[a.step](a)


if __name__ == "__main__":
    main()

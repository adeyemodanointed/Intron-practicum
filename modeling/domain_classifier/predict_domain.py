"""Label any CSV or JSONL file with the multilingual domain classifier.

    python predict_domain.py in.csv out.csv --text-col Text --lang swa
    python predict_domain.py in.jsonl out.jsonl --text-col raw_text --lang-col language

Adds: domains (one or two of agriculture/finance/health/law/telecoms joined by "|", or "other"),
primary_domain, and p_<domain> probabilities. Language codes: swa, kin, pcm (others work, with no language prior).
"""
import argparse, csv, json, os, sys

import joblib

from domain_model import DOMAINS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--text-col", default="text"); ap.add_argument("--lang-col"); ap.add_argument("--lang")
    ap.add_argument("--model", default=os.path.join(HERE, "model", "domain_classifier.joblib"))
    a = ap.parse_args()
    if not (a.lang or a.lang_col):
        sys.exit("give --lang (one language for every row) or --lang-col")
    jsonl = a.inp.endswith(".jsonl")
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")] if jsonl else list(csv.DictReader(open(a.inp, encoding="utf-8-sig")))
    m = joblib.load(a.model)
    texts = [str(r[a.text_col]) for r in rows]
    langs = [a.lang or r[a.lang_col] for r in rows]
    P = m.proba(texts, langs)
    for r, labs, p in zip(rows, m.decide(P, langs), P):
        r["domains"] = "|".join(labs); r["primary_domain"] = labs[0]
        for d, x in zip(DOMAINS, p):
            r[f"p_{d}"] = round(float(x), 3)
    if jsonl or a.out.endswith(".jsonl"):
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("labelled", len(rows), "rows ->", a.out)


if __name__ == "__main__":
    main()

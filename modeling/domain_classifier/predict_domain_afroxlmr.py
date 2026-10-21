"""Label any CSV or JSONL file with the fine-tuned AfroXLMR domain classifier.

    python predict_domain_afroxlmr.py in.csv out.csv --model-dir afroxlmr_out/model --text-col Text --lang swa
    python predict_domain_afroxlmr.py in.jsonl out.jsonl --model-dir afroxlmr_out/model --text-col raw_text --lang-col language

Adds: domains (one or two of agriculture/finance/health/law/telecoms joined by "|", or "other"), primary_domain,
p_<domain>. Languages with their own thresholds use them; any other language (e.g. ibo, yor) uses the shared ones.
Runs on GPU if available, otherwise CPU (slower).
"""
import argparse, csv, json, sys

import torch

from domain_model import DOMAINS
from train_afroxlmr_domain import decide, predict_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text-col", default="text"); ap.add_argument("--lang-col"); ap.add_argument("--lang")
    ap.add_argument("--max-len", type=int, default=256)
    a = ap.parse_args()
    if not (a.lang or a.lang_col):
        sys.exit("give --lang (one language for every row) or --lang-col")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_dir).to(device)
    th = json.load(open(f"{a.model_dir}/thresholds.json"))
    jsonl = a.inp.endswith(".jsonl")
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")] if jsonl else list(csv.DictReader(open(a.inp, encoding="utf-8-sig")))
    langs = [a.lang or r[a.lang_col] for r in rows]
    ids = tok([str(r[a.text_col]) for r in rows], truncation=True, max_length=a.max_len)["input_ids"]
    P = predict_proba(model, ids, tok.pad_token_id, device)
    for r, labs, p in zip(rows, decide(P, langs, th), P):
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

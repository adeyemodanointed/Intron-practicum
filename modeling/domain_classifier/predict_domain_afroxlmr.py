"""Label any CSV or JSONL file with the fine-tuned AfroXLMR domain classifier (one or more AfroXLMR models, optionally
mixed with TF-IDF, as packaged by cv_collect.py --package).

    python predict_domain_afroxlmr.py in.csv out.csv --package final_model --text-col Text --lang swa
    python predict_domain_afroxlmr.py in.jsonl out.jsonl --package final_model --text-col raw_text --lang-col language --per-sentence
    python predict_domain_afroxlmr.py in.csv out.csv --model-dir afroxlmr_out/model --lang kin      # one model + its thresholds.json

Adds to every row:
  domains          every domain the text belongs to, highest first, joined by "|" (e.g. "finance|telecoms"), or "other"
  primary_domain   the first of them
  is_multi_domain  True when the text has two or more domains
  p_<label>        probability for each output (agriculture, finance, health, law, telecoms and, for six-output models, other)
--per-sentence also labels each sentence and adds `sentence_domains` (JSON list of [sentence, domains]); a domain that
labels at least --sentence-share of the sentences is added to the text's domains (up to --max-labels), so a passage
that moves from one topic to another gets both.
Languages with their own thresholds use them; any other language uses the shared ones. GPU if available, else CPU.
"""
import argparse, csv, json, os, re, sys

import numpy as np
import torch

from domain_model import DOMAINS, LABELS, OTHER, decide_labels
from train_afroxlmr_domain import predict_proba


def load_predictor(a):
    if a.package:
        cfg = json.load(open(os.path.join(a.package, "predictor.json")))
        base = a.package
    else:
        cfg = {"models": [a.model_dir], "tfidf": None, "tfidf_weight": 0.0,
               "thresholds": json.load(open(os.path.join(a.model_dir, "thresholds.json")))}
        base = ""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for d in cfg["models"]:
        d = os.path.join(base, d)
        models.append((AutoTokenizer.from_pretrained(d), AutoModelForSequenceClassification.from_pretrained(d).to(device)))
    tfidf = None
    if cfg.get("tfidf"):
        import joblib
        tfidf = joblib.load(os.path.join(base, cfg["tfidf"]))
    width = min([m.config.num_labels for _, m in models] + ([len(tfidf.outputs())] if tfidf else []))

    def proba(texts, langs):
        P = np.mean([predict_proba(m, tok(texts, truncation=True, max_length=a.max_len)["input_ids"], tok.pad_token_id, device)[:, :width]
                     for tok, m in models], axis=0)
        if tfidf:
            w = cfg["tfidf_weight"]
            P = (1 - w) * P + w * tfidf.proba(texts, langs)[:, :width]
        return P
    return proba, cfg["thresholds"], LABELS[:width] if width > len(DOMAINS) else DOMAINS


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", t).strip()) if len(s.split()) >= 3] or [t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--package", help="directory with predictor.json (from cv_collect.py --package)")
    ap.add_argument("--model-dir", help="one model directory with thresholds.json")
    ap.add_argument("--text-col", default="text"); ap.add_argument("--lang-col"); ap.add_argument("--lang")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-labels", type=int, default=2)
    ap.add_argument("--per-sentence", action="store_true")
    ap.add_argument("--sentence-share", type=float, default=0.3)
    a = ap.parse_args()
    if not (a.lang or a.lang_col):
        sys.exit("give --lang (one language for every row) or --lang-col")
    if not (a.package or a.model_dir):
        sys.exit("give --package or --model-dir")
    proba, th, cols = load_predictor(a)
    jsonl = a.inp.endswith(".jsonl")
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")] if jsonl else list(csv.DictReader(open(a.inp, encoding="utf-8-sig")))
    texts = [str(r[a.text_col]) for r in rows]
    langs = [a.lang or r[a.lang_col] for r in rows]
    P = proba(texts, langs)
    labels = decide_labels(P, langs, th, a.max_labels)
    if a.per_sentence:
        sents = [sentences(t) for t in texts]
        flat = [s for ss in sents for s in ss]
        flat_lang = [l for l, ss in zip(langs, sents) for _ in ss]
        flat_lab = decide_labels(proba(flat, flat_lang), flat_lang, th, a.max_labels)
        pos = 0
        for i, ss in enumerate(sents):
            sl = flat_lab[pos: pos + len(ss)]; pos += len(ss)
            rows[i]["sentence_domains"] = json.dumps([[s, "|".join(l)] for s, l in zip(ss, sl)], ensure_ascii=False)
            share = {d: sum(d in l for l in sl) / len(ss) for d in DOMAINS}
            extra = [d for d in sorted(DOMAINS, key=lambda d: -P[i][DOMAINS.index(d)]) if share[d] >= a.sentence_share]
            merged = [d for d in labels[i] if d != OTHER] + [d for d in extra if d not in labels[i]]
            labels[i] = merged[: a.max_labels] or [OTHER]
    for r, labs, p in zip(rows, labels, P):
        r["domains"] = "|".join(labs); r["primary_domain"] = labs[0]; r["is_multi_domain"] = len(labs) > 1
        for c, x in zip(cols, p):
            r[f"p_{c}"] = round(float(x), 3)
    if jsonl or a.out.endswith(".jsonl"):
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("labelled", len(rows), "rows ->", a.out, "| multi-domain:", sum(len(l) > 1 for l in labels))


if __name__ == "__main__":
    main()

"""Swahili question-answer generation from licence-clear domain passages (D2).
Pipeline: select passages from Data.csv (train split) -> prompt an instruction-tuned LLM (default Gemma 3, any HF chat model) to write questions that the passage answers
-> automatic filters -> JSONL with the source passage id. Every pair is auto-generated and UNREVIEWED (review_status = auto_generated_unreviewed);
a native-speaker sample must be reviewed before any pair is called verified.

Usage (free Colab/Kaggle T4: add --load-4bit; the run resumes if the session stops)
  python generate_qa_swahili.py --data ../Data.csv --ids ../Data_row_ids.csv --per-domain 300 --dry-run          # select passages, write prompts, no model
  python generate_qa_swahili.py --data ../Data.csv --ids ../Data_row_ids.csv --per-domain 300 --model google/gemma-3-4b-it   # needs a GPU and HF access to the model
Outputs: qa_source_passages.jsonl, qa_pairs_swahili.jsonl (appended, resumable), qa_filter_log.json
"""
import argparse, csv, json, os, random, re, collections
SW = set("na ya wa kwa ni katika za la cha hii huu kuwa pia au ambayo hiyo vya kama hadi kutoka huo hizo yake wake lake ili sana kuhusu lakini kwamba pamoja".split())
DOMAIN_SW = {"agriculture": "kilimo", "finance": "fedha na uchumi", "health": "afya", "law": "sheria", "telecoms": "mawasiliano na simu"}
PROMPT = """Wewe ni mwandishi wa maswali ya mtihani kwa lugha ya Kiswahili sanifu. Soma kifungu kifuatacho (mada: {topic}).

KIFUNGU:
{passage}

Andika maswali {n} ya Kiswahili ambayo majibu yake yanapatikana wazi katika kifungu hicho tu. Kila swali liwe kamili (lieleweke bila kusoma kifungu), liishe na alama ya kuuliza (?), na jibu liwe fupi (sentensi moja) na litumie maneno ya kifungu.
Toa matokeo kama JSON tu, orodha ya vitu vyenye "question" na "answer". Usiongeze maelezo mengine."""
def words(t): return re.findall(r"[\w']+", t.lower())
def is_sw(t):
    w = words(t); return bool(w) and sum(x in SW for x in w) / len(w) >= 0.04
def supported(answer, passage):
    a = [x for x in words(answer) if len(x) > 3]; p = set(words(passage))
    return bool(a) and sum(x in p for x in a) / len(a) >= 0.6
def select(args):
    ids = [r["id"] for r in csv.DictReader(open(args.ids, encoding="utf-8")) if r["file"] == "Data.csv"]
    rows = list(csv.DictReader(open(args.data, encoding="utf-8-sig")))
    assert len(ids) == len(rows), "Data_row_ids.csv does not match Data.csv"
    rng = random.Random(args.seed); by = collections.defaultdict(list)
    for rid, r in zip(ids, rows):
        n = len(r["Text"].split())
        if r["Split"] == "train" and args.min_words <= n <= args.max_words and is_sw(r["Text"]): by[r["Domain"]].append({"id": rid, "domain": r["Domain"], "source": r["Source"], "license": r["License"], "text": r["Text"]})
    out = []
    for d, lst in sorted(by.items()): rng.shuffle(lst); out += lst[: args.per_domain]
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--per-domain", type=int, default=300)
    ap.add_argument("--questions", type=int, default=3); ap.add_argument("--min-words", type=int, default=60); ap.add_argument("--max-words", type=int, default=200)
    ap.add_argument("--model", default="google/gemma-3-4b-it"); ap.add_argument("--seed", type=int, default=13); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--out-dir", default="."); ap.add_argument("--load-4bit", action="store_true", help="4-bit weights (bitsandbytes): fits a free Colab/Kaggle T4"); ap.add_argument("--batch", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    sel = select(a)
    with open(os.path.join(a.out_dir, "qa_source_passages.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for s in sel: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("selected", len(sel), dict(collections.Counter(s["domain"] for s in sel)))
    if a.dry_run:
        print(PROMPT.format(topic=DOMAIN_SW[sel[0]["domain"]], passage=sel[0]["text"], n=a.questions)); return
    import torch
    from transformers import pipeline
    kw = {"device_map": "auto"}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16   # T4 has no native bf16
    gen = pipeline("text-generation", model=a.model, **kw)
    qpath = os.path.join(a.out_dir, "qa_pairs_swahili.jsonl"); done = set()
    if os.path.exists(qpath): done = {json.loads(l)["passage_id"] for l in open(qpath, encoding="utf-8")}
    log = collections.Counter()
    with open(qpath, "a", encoding="utf-8", newline="\n") as out:
        for i, s in enumerate(sel):
            if s["id"] in done: continue
            msg = [{"role": "user", "content": PROMPT.format(topic=DOMAIN_SW[s["domain"]], passage=s["text"], n=a.questions)}]
            txt = gen(msg, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=0.4, top_p=0.9)[0]["generated_text"][-1]["content"]
            m = re.search(r"\[.*\]", txt, re.S)
            try: items = json.loads(m.group(0)) if m else []
            except Exception: items = []; log["bad_json"] += 1
            kept = 0
            for it in items:
                q, ans = str(it.get("question", "")).strip(), str(it.get("answer", "")).strip()
                if not q.endswith("?"): log["no_question_mark"] += 1; continue
                if not is_sw(q) or not is_sw(ans): log["not_swahili"] += 1; continue
                if not supported(ans, s["text"]): log["answer_not_in_passage"] += 1; continue
                out.write(json.dumps({"passage_id": s["id"], "language": "swa", "domain": s["domain"], "question": q, "answer": ans, "passage": s["text"], "source_dataset": s["source"], "license": s["license"],
                                      "generator": a.model, "review_status": "auto_generated_unreviewed"}, ensure_ascii=False) + "\n"); kept += 1; log["kept"] += 1
            out.flush()
            if i % 25 == 0: print(i, "of", len(sel), dict(log), flush=True)
    json.dump(dict(log), open(os.path.join(a.out_dir, "qa_filter_log.json"), "w"), indent=1); print("done", dict(log))
if __name__ == "__main__": main()

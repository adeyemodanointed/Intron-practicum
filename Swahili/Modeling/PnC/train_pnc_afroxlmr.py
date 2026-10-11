"""Swahili punctuation and capitalisation restoration with a token classifier (default Davlan/afro-xlmr-base).
Input to the model: lowercase words with no punctuation (Text_No_Punct style). Two heads per word: punctuation after the word (O, COMMA, PERIOD, QUESTION, EXCL, OTHER)
and capitalisation of the word (LOWER, TITLE, UPPER, MIXED). Labels come from the reference text with the same rules as evaluate_pnc.py.
Train: Data.csv rows with Split == train (optionally capped by --max-train-words). Predict: Evaluation.csv rows -> Hypothesis column. Then run evaluate_pnc.py.
Limitation: MIXED words are restored as TITLE; punctuation before a word (opening quotes/brackets) is not modelled. Questions are rare in the data (about 0.08% of words), so check QUESTION recall.
Usage: python train_pnc_afroxlmr.py --data ../../Data.csv --eval ../../Evaluation.csv --epochs 2 --max-train-words 300000
Smoke test on CPU: add --model hf-internal-testing/tiny-random-xlm-roberta --max-train-words 3000 --epochs 1 --batch 8
"""
import argparse, csv, os, sys, random, json, collections, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Evaluation"))
from evaluate_pnc import labels as ref_labels
PL = ["O", "COMMA", "PERIOD", "QUESTION", "EXCL", "OTHER"]; CL = ["LOWER", "TITLE", "UPPER", "MIXED"]
class M(nn.Module):
    def __init__(self, name):
        super().__init__(); self.enc = AutoModel.from_pretrained(name); h = self.enc.config.hidden_size; self.p = nn.Linear(h, len(PL)); self.c = nn.Linear(h, len(CL)); self.drop = nn.Dropout(0.1)
    def forward(self, **kw):
        x = self.drop(self.enc(**kw).last_hidden_state); return self.p(x), self.c(x)
def encode(tok, items, maxlen):
    """items: list of (words, punct_ids or None, cap_ids or None). Words are split into windows of at most maxlen subword tokens; label goes on the first subword."""
    out = []
    for words, pl, cl in items:
        i = 0
        while i < len(words):
            j, n = i, 0
            while j < len(words) and n + len(tok.tokenize(words[j]) or ["_"]) + 2 <= maxlen: n += len(tok.tokenize(words[j]) or ["_"]); j += 1
            j = max(j, i + 1)
            enc = tok(words[i:j], is_split_into_words=True, truncation=True, max_length=maxlen, return_tensors=None)
            wid = enc.word_ids(); first = [None] * len(wid); seen = set()
            for k, w in enumerate(wid):
                if w is not None and w not in seen: seen.add(w); first[k] = w
            out.append({"input_ids": enc["input_ids"], "first": first, "span": (i, j), "pl": pl[i:j] if pl else None, "cl": cl[i:j] if cl else None}); i = j
    return out
def batchify(tok, ex, bs, shuffle):
    idx = list(range(len(ex)))
    if shuffle: random.shuffle(idx)
    for s in range(0, len(idx), bs):
        b = [ex[k] for k in idx[s:s + bs]]; L = max(len(e["input_ids"]) for e in b); pad = tok.pad_token_id
        ids = torch.tensor([e["input_ids"] + [pad] * (L - len(e["input_ids"])) for e in b]); att = (ids != pad).long()
        yp = torch.full(ids.shape, -100); yc = torch.full(ids.shape, -100)
        if b[0]["pl"] is not None:
            for r, e in enumerate(b):
                for k, w in enumerate(e["first"]):
                    if w is not None: yp[r, k] = e["pl"][w]; yc[r, k] = e["cl"][w]
        yield b, ids, att, yp, yc
def rebuild(words, pp, cp):
    out = []
    for w, p, c in zip(words, pp, cp):
        w = w.upper() if c == "UPPER" else (w[:1].upper() + w[1:] if c in ("TITLE", "MIXED") else w)
        out.append(w + {"O": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?", "EXCL": "!", "OTHER": ";"}[p])
    return " ".join(out)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--model", default="Davlan/afro-xlmr-base")
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--max-train-words", type=int, default=0); ap.add_argument("--seed", type=int, default=13); ap.add_argument("--save", default="")
    a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; tok = AutoTokenizer.from_pretrained(a.model); model = M(a.model).to(dev)
    rows = [r for r in csv.DictReader(open(a.data, encoding="utf-8-sig")) if r["Split"] == "train"]; random.shuffle(rows)
    items, total = [], 0
    for r in rows:
        lab = ref_labels(r["Text"]); 
        if not lab: continue
        items.append(([w.lower() for w, _, _ in lab], [PL.index(p) for _, p, _ in lab], [CL.index(c) for _, _, c in lab])); total += len(lab)
        if a.max_train_words and total >= a.max_train_words: break
    print("training words", total, "passages", len(items), "device", dev, flush=True)
    ex = encode(tok, items, a.maxlen); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); lossf = nn.CrossEntropyLoss(ignore_index=-100)
    steps = a.epochs * ((len(ex) + a.batch - 1) // a.batch); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, steps))); st = 0; scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    for ep in range(a.epochs):
        model.train(); run = 0
        for b, ids, att, yp, yc in batchify(tok, ex, a.batch, True):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                lp, lc = model(input_ids=ids.to(dev), attention_mask=att.to(dev))
            loss = lossf(lp.float().view(-1, len(PL)), yp.to(dev).view(-1)) + lossf(lc.float().view(-1, len(CL)), yc.to(dev).view(-1))
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); st += 1; run += loss.item()
            if st % 50 == 0: print(f"epoch {ep} step {st}/{steps} loss {run/50:.4f}", flush=True); run = 0
    model.eval(); E = list(csv.DictReader(open(a.eval, encoding="utf-8-sig")))
    for r in E:
        words = r["Text_No_Punct"].split(); pp, cp = ["O"] * len(words), ["LOWER"] * len(words)
        exs = encode(tok, [(words, None, None)], a.maxlen)
        with torch.no_grad():
            for b, ids, att, _, _ in batchify(tok, exs, 32, False):
                lp, lc = model(input_ids=ids.to(dev), attention_mask=att.to(dev))
                for k, e in enumerate(b):
                    for pos, w in enumerate(e["first"]):
                        if w is not None: gi = e["span"][0] + w; pp[gi] = PL[lp[k, pos].argmax().item()]; cp[gi] = CL[lc[k, pos].argmax().item()]
        r["Hypothesis"] = rebuild(words, pp, cp)
    with open(a.eval, "w", encoding="utf-8-sig", newline="") as f: w = csv.DictWriter(f, fieldnames=list(E[0].keys())); w.writeheader(); w.writerows(E)
    if a.save: model.enc.save_pretrained(a.save); tok.save_pretrained(a.save)
    print("wrote Hypothesis for", len(E), "rows; now run Evaluation/evaluate_pnc.py")
if __name__ == "__main__": main()

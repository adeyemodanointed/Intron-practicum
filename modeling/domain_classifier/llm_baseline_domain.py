"""LLM baselines for the domain task: ask an instruction-tuned LLM for the domains of each text (same six labels, one or
two domains, or other) and write predictions that evaluate_domain.py scores like any other system.

    # open models on a GPU (Kaggle T4 x2): Hugging Face transformers, 4-bit for models over 9B parameters
    python llm_baseline_domain.py --backend hf --model /kaggle/input/gemma-3/transformers/gemma-3-12b-it/1 --shots 0 --out llm/gemma3_12b_0shot
    # an API model; the key is read from OPENAI_API_KEY / GEMINI_API_KEY in your own shell and never written anywhere
    python llm_baseline_domain.py --backend openai --model gpt-6.1-sol --shots 6 --price 2 10 --out results/llm/gpt61sol_6shot
    python llm_baseline_domain.py --backend gemini --model gemini-4-argon --shots 0 --price 2 10 --out results/llm/gemini4argon_0shot
    # label other texts (e.g. multi-domain candidates from make_multi_candidates.py)
    python llm_baseline_domain.py --backend hf --model ... --input data_v3/multi_candidates.jsonl --all-rows --out llm/multi_candidates

Input rows: --input (default data_v3/domain_eval.jsonl); by default only human-labelled rows are sent (--all-rows for all).
Prompt: one-line definitions of the six labels, the language name, the text (first --max-words words), and a JSON answer
{"domains": [...]}, most relevant first, at most two. --shots 6 adds one example per label in the same language, taken
from domain_train.jsonl (never from the rows being scored). Greedy decoding / temperature 0.
Answers that are not valid JSON are parsed for label words; anything unreadable counts as `other` (counted in run_info).
Writes test_predictions.csv (evaluate_domain.py format, plus the raw answer), responses.jsonl (cache: a rerun skips rows
already answered) and run_info.json (model, shots, rows, seconds per text, tokens, parse failures).
"""
import argparse, collections, csv, json, os, random, re, sys, time

from domain_model import DOMAINS, LABELS, OTHER, norm_labels

HERE = os.path.dirname(os.path.abspath(__file__))
LANG_NAMES = {"swa": "Swahili", "kin": "Kinyarwanda", "pcm": "Nigerian Pidgin", "yor": "Yoruba", "ibo": "Igbo", "hau": "Hausa"}
DEFINITIONS = {
    "agriculture": "farming, crops, livestock, fishing, farm inputs, harvests and food production",
    "finance": "money, banking, loans, savings, payments, insurance, taxes, prices, business and the economy",
    "health": "illness, symptoms, treatment, medicine, hospitals, nutrition, pregnancy and public health",
    "law": "laws, rights, courts, police and crime, contracts, regulations and legal procedures",
    "telecoms": "mobile phones, networks, airtime, data bundles, internet, SIM cards and broadcasting",
    "other": "none of the five domains, e.g. sport, entertainment, religion, general politics or everyday news",
}
SYSTEM = ("You classify short texts in African languages into domains. Domains:\n"
          + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
          + "\nGive one domain, or two if the text is clearly about both. Use \"other\" only when none of the five domains fits. "
            "Answer with JSON only, most relevant domain first, for example {\"domains\": [\"health\"]} or {\"domains\": [\"finance\", \"telecoms\"]}.")


def clip(t, n):
    w = str(t).split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def user_msg(r, max_words):
    return f"Language: {LANG_NAMES.get(r['language'], r['language'])}\nText: {clip(r['text'], max_words)}\nDomains (JSON):"


def few_shot(pool, lang, seed, max_words):
    """One training example per label in this language (labels without examples are skipped)."""
    rng = random.Random(seed)
    msgs = []
    by = collections.defaultdict(list)
    for r in pool:
        if r["language"] == lang and len(r["labels"]) == 1:
            by[r["labels"][0]].append(r)
    for lab in LABELS:
        if by[lab]:
            ex = rng.choice(by[lab])
            msgs += [{"role": "user", "content": user_msg(ex, max_words)}, {"role": "assistant", "content": json.dumps({"domains": [lab]})}]
    return msgs


def parse(answer):
    """-> (labels, ok). JSON first, then label words in order of appearance."""
    m = re.search(r"\{.*?\}", answer, re.S)
    if m:
        try:
            raw = json.loads(m.group(0)).get("domains", [])
            raw = [raw] if isinstance(raw, str) else list(raw)
            labs = []
            for x in raw:   # a label outside the list (e.g. "sports") means none of the five domains: other
                try:
                    labs += norm_labels([x])
                except ValueError:
                    labs.append(OTHER)
            labs = list(dict.fromkeys(labs))
            if labs:
                return ([l for l in labs if l != OTHER][:2] or [OTHER]), True
        except (ValueError, AttributeError, TypeError):
            pass
    found = sorted((m.start(), l) for l in LABELS for m in [re.search(r"\b" + l + r"\b", answer.lower())] if m)
    labs = [l for _, l in found]
    labs = [l for l in labs if l != OTHER][:2] or labs[:1]
    return (labs, False) if labs else ([OTHER], False)


# ---------------------------------------------------------------- backends
class HF:
    def __init__(self, a):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = getattr(torch, a.dtype)
        kw = {"device_map": "auto", "torch_dtype": dtype} if torch.cuda.is_available() else {}
        if a.load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
        except (ValueError, KeyError):   # multimodal checkpoints (e.g. Gemma 3 4B and up)
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(a.model, **kw).eval()
        self.batch, self.max_new = a.batch, a.max_new_tokens

    def run(self, convs):
        out = []
        for i in range(0, len(convs), self.batch):
            chunk = convs[i:i + self.batch]
            texts = [self.chat_text(c) for c in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
            for g in gen[:, enc["input_ids"].shape[1]:]:
                out.append((self.tok.decode(g, skip_special_tokens=True), int(enc["input_ids"].shape[1]), int((g != self.tok.pad_token_id).sum())))
        return out

    def chat_text(self, conv):
        try:
            return self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        except Exception:   # templates without a system role (e.g. Gemma 2): fold the system text into the first user turn
            conv = [dict(conv[1], content=conv[0]["content"] + "\n\n" + conv[1]["content"])] + conv[2:]
            return self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


class API:
    """OpenAI or Gemini over HTTPS; the key comes from the environment only."""
    def __init__(self, a):
        import requests
        self.requests, self.a = requests, a
        var = "OPENAI_API_KEY" if a.backend == "openai" else "GEMINI_API_KEY"
        self.key = os.environ.get(var)
        if not self.key:
            sys.exit(f"set {var} in your shell first")
        import threading
        self.lock, self.next_at, self.variant, self.fatal = threading.Lock(), 0.0, 0, None

    def throttle(self):
        """At most --rpm requests per minute across all workers (free API tiers)."""
        if not self.a.rpm:
            return
        with self.lock:
            now = time.time()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + 60.0 / self.a.rpm
        time.sleep(wait)

    # Request variants, tried in order until the API accepts one (newer models reject some parameters): the first ones
    # switch reasoning/thinking off or to its minimum, because thinking tokens are billed as output.
    OPENAI_VARIANTS = [{"reasoning_effort": "none"}, {"reasoning_effort": "minimal"}, {"reasoning_effort": "low"}, {"temperature": 0}, {}]
    GEMINI_VARIANTS = [{"thinkingBudget": 0}, {"thinkingLevel": "minimal"}, {"thinkingLevel": "low"}, None]

    def rejected(self, r):
        """True when this request variant's parameters were refused (HTTP 400); raises on key/access/model errors."""
        if r.status_code in (401, 403, 404) or (r.status_code == 400 and re.search(r"API[_ ]KEY|api key", r.text, re.I)):
            raise PermissionError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code == 400:
            self.last_400 = r.text[:300]
            return True
        return False

    def request(self, conv, v):
        a = self.a
        if a.backend == "openai":
            body = {"model": a.model, "messages": conv, "max_completion_tokens": a.max_api_tokens,
                    "response_format": {"type": "json_object"}, **self.OPENAI_VARIANTS[v]}
            r = self.requests.post("https://api.openai.com/v1/chat/completions", timeout=120,
                                   headers={"Authorization": f"Bearer {self.key}"}, json=body)
            if self.rejected(r):
                return None
            r.raise_for_status(); j = r.json()
            u = j.get("usage", {})
            return j["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        gen = {"temperature": 0, "maxOutputTokens": a.max_api_tokens, "responseMimeType": "application/json"}
        if self.GEMINI_VARIANTS[v] is not None:
            gen["thinkingConfig"] = self.GEMINI_VARIANTS[v]
        body = {"systemInstruction": {"parts": [{"text": conv[0]["content"]}]},
                "contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in conv[1:]],
                "generationConfig": gen}
        r = self.requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{a.model}:generateContent", timeout=120,
                               headers={"x-goog-api-key": self.key}, json=body)
        if self.rejected(r):
            return None
        r.raise_for_status(); j = r.json()
        text = "".join(p.get("text", "") for p in j["candidates"][0].get("content", {}).get("parts", []) if not p.get("thought"))
        u = j.get("usageMetadata", {})
        return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)

    def one(self, conv):
        n_var = len(self.OPENAI_VARIANTS if self.a.backend == "openai" else self.GEMINI_VARIANTS)
        attempt = 0
        while attempt < 8 and self.variant < n_var:
            self.throttle()
            v = self.variant
            try:
                out = self.request(conv, v)
                if out is None:   # parameter rejected: move every worker to the next variant
                    with self.lock:
                        self.variant = max(self.variant, v + 1)
                    print(f"  request variant {v} rejected; trying {v + 1}", flush=True)
                    continue
                return out
            except PermissionError as e:   # wrong key, no access or unknown model: retrying will not help
                self.fatal = str(e)
                return "", 0, 0
            except Exception as e:   # rate limits and transient errors: back off and retry
                wait = min(60, 2 ** attempt); attempt += 1
                print(f"  retry in {wait}s: {type(e).__name__} {str(e)[:120]}", flush=True)
                time.sleep(wait)
        if self.variant >= n_var:
            self.fatal = f"every request variant was refused; last answer: {getattr(self, 'last_400', '')}"
        return "", 0, 0

    def run(self, convs):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.a.workers) as ex:
            return list(ex.map(self.one, convs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("hf", "openai", "gemini"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default=os.path.join(HERE, "data_v3", "domain_eval.jsonl"))
    ap.add_argument("--train", default=os.path.join(HERE, "data_v3", "domain_train.jsonl"), help="few-shot examples come from here")
    ap.add_argument("--all-rows", action="store_true", help="also send rows that are not human-labelled")
    ap.add_argument("--shots", type=int, choices=(0, 6), default=0)
    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=0, help="API: at most this many requests per minute (0 = no limit)")
    ap.add_argument("--max-api-tokens", type=int, default=2048, help="API: output budget per answer, thinking included")
    ap.add_argument("--price", type=float, nargs=2, metavar=("IN", "OUT"), help="API: USD per 1M input and output tokens, for the cost in run_info")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    ap.add_argument("--limit", type=int, default=0, help="quick check: only this many rows")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    if not a.all_rows:
        rows = [r for r in rows if r.get("label_source", "gold").startswith("gold")]
    if a.limit:
        rows = random.Random(a.seed).sample(rows, min(a.limit, len(rows)))
    os.makedirs(a.out, exist_ok=True)
    cache_path = os.path.join(a.out, "responses.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path, encoding="utf-8"):
            c = json.loads(l)
            if c["answer"].strip():   # empty answers (API errors) are sent again
                cache[c["id"]] = c
    pool = [json.loads(l) for l in open(a.train, encoding="utf-8")] if a.shots else []
    shots = {l: few_shot(pool, l, a.seed, 60) for l in {r["language"] for r in rows}} if a.shots else {}
    todo = [r for r in rows if r["id"] not in cache]
    print(f"{a.backend} {a.model}: {len(rows)} rows, {len(todo)} to send, {a.shots}-shot", flush=True)
    backend = HF(a) if a.backend == "hf" else API(a)
    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as f:
        step = 256
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            convs = [[{"role": "system", "content": SYSTEM}] + shots.get(r["language"], []) + [{"role": "user", "content": user_msg(r, a.max_words)}] for r in chunk]
            for r, (ans, pin, pout) in zip(chunk, backend.run(convs)):
                c = {"id": r["id"], "answer": ans, "prompt_tokens": pin, "output_tokens": pout}
                cache[r["id"]] = c; f.write(json.dumps(c, ensure_ascii=False) + "\n")
            f.flush()
            if getattr(backend, "fatal", None):
                sys.exit(f"stopped: the API refused the request ({backend.fatal}). Check the key, billing and model name.")
            el = time.time() - t0
            print(f"  {i + len(chunk)}/{len(todo)} sent, {el / 60:.1f} min, about {el / (i + len(chunk)) * (len(todo) - i - len(chunk)) / 60:.0f} min left", flush=True)
    fails = 0
    with open(os.path.join(a.out, "test_predictions.csv"), "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["id", "language", "fold", "source", "human_labels", "predicted_labels", "correct_primary", "exact", "parsed", "answer", "text"])
        for r in rows:
            labs, ok = parse(cache[r["id"]]["answer"])
            fails += not ok
            w.writerow([r["id"], r["language"], r.get("fold", ""), r.get("source", ""), "|".join(r["labels"]), "|".join(labs), labs[0] in r["labels"],
                        set(labs) == set(r["labels"]), ok, cache[r["id"]]["answer"].replace("\n", " ")[:300], r["text"]])
    sent = [cache[r["id"]] for r in rows]
    info = {"backend": a.backend, "model": a.model, "shots": a.shots, "rows": len(rows), "parse_failures": fails,
            "seconds_this_run": round(time.time() - t0, 1), "seconds_per_text_this_run": round((time.time() - t0) / max(1, len(todo)), 3),
            "prompt_tokens": sum(c["prompt_tokens"] for c in sent), "output_tokens": sum(c["output_tokens"] for c in sent)}
    if a.price:
        info["usd"] = round((info["prompt_tokens"] * a.price[0] + info["output_tokens"] * a.price[1]) / 1e6, 2)
    if a.backend != "hf":
        vs = API.OPENAI_VARIANTS if a.backend == "openai" else API.GEMINI_VARIANTS
        info["request_variant"] = vs[min(backend.variant, len(vs) - 1)]
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    print(info)


if __name__ == "__main__":
    main()

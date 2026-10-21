"""Build one train/dev/test set for the multilingual domain classifier.

Record (JSONL): id, language, text, labels (list: one or two domains, or ["other"]), split, source, label_source.
label_source: gold:* = a person or a human dataset label assigned it; weak:* = rule, keyword or source default.

Languages available now: swa, kin, pcm. ibo and yor can be added with --extra (CSV or JSONL with
columns language, text, labels, split; labels like "finance" / "finance|law" / "other").

Splits
  train  every weak/silver row, plus gold training rows (pcm MasakhaNEWS labels)
  dev    one third of each language's human test set (stratified, seed 13) + swa human-kept review passages;
         used only to tune the per-domain thresholds
  test   the other two thirds of each human test set (pcm: its own human test split)
Leakage: a training row is dropped if it matches a dev/test text exactly or shares 3+ word 8-grams with one.

Usage: python build_domain_data.py [--extra more.csv ...]
"""
import argparse, collections, csv, io, json, os, random, re, zipfile

import openpyxl
import pandas as pd

from domain_model import DOMAINS, OTHER, norm_labels

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SF = os.path.join(ROOT, "outputs", "deliverables-2026-10-04", "shared_format")
EXT = os.path.join(ROOT, "outputs", "deliverables-2026-10-04", "external")
IN = {
    "swa_data": os.path.join(SF, "Data.csv"),
    "swa_news_tier": os.path.join(SF, "Data_news_tier.csv"),
    "swa_other_news": os.path.join(EXT, "mollel_news.parquet"),
    "swa_gold_xlsx": os.path.join(ROOT, "swahili_360_annotation.xlsx"),
    "swa_gold_labels": os.path.join(ROOT, "swahili_domain_column_B362.txt"),
    "kin_zip": os.path.join(HERE, "inputs", "kinyarwanda_classifier_final.zip"),
    "pcm_zip": os.path.join(HERE, "inputs", "pcm_results.zip"),
}
OUT = os.path.join(HERE, "data")
SEED = 13

# Swahili domain stems (same lexicon as build_questions.py / process_wikipedia.py); used only to keep
# domain-looking news chunks out of the Swahili `other` examples.
SW_LEX = {
    "law": "sheri mahakam jaji hakim katiba haki mkataba wakili kesi hukumu adhabu uhalifu mashtaka bunge uchaguzi polisi gereza azimio itifaki mwanasheria kanuni".split(),
    "telecoms": "simu mawasiliano intaneti mtandao redio televisheni setilaiti satelaiti utangazaji posta masafa ujumbe waya mnara wireless broadband gsm mpokeaji".split(),
    "finance": "benki fedha pesa uchumi biashara bima mkopo soko hisa kodi bei mapato faida uwekezaji mtaji sarafu shilingi dola malipo akiba mteja kampuni".split(),
    "agriculture": "kilimo mazao mkulima wakulima shamba mbegu udongo mifugo ng'ombe mbolea mavuno zao mmea mimea ufugaji kuku maziwa umwagiliaji wadudu mvua".split(),
    "health": "afya ugonjwa magonjwa dawa tiba daktari hospitali chanjo mgonjwa virusi bakteria dalili matibabu lishe mimba damu kinga maambukizi".split(),
}
SW_RX = [re.compile(r"\b" + re.escape(s) + r"\w*", re.I) for v in SW_LEX.values() for s in v]
# Pidgin agriculture keywords (from langs/pcm.py). The Pidgin pipeline treated agriculture as `other`,
# so keyword-labelled `other` rows that are clearly about agriculture are relabelled here.
PCM_AGRI = ("farm farms farmer farmers farming agric agriculture agricultural crop crops harvest rice cassava maize "
            "corn yam cocoa fertiliser fertilizer livestock cattle cow cows poultry goat herders herdsmen seed seeds "
            "irrigation planting soil grains pesticide tomato tomatoes").split()
PCM_AGRI_RX = re.compile(r"\b(" + "|".join(map(re.escape, PCM_AGRI)) + r")\b", re.I)


def sw_hits(t):
    return sum(1 for r in SW_RX if r.search(t))


def rec(lang, text, labels, split, source, label_source, rid):
    return {"id": rid, "language": lang, "text": re.sub(r"\s+", " ", str(text)).strip(), "labels": norm_labels(labels),
            "split": split, "source": source, "label_source": label_source}


def chunks(article, lo=30, hi=80, take=2):
    """Whole-sentence chunks of lo..hi words (the human test sets are sentence/short-passage length)."""
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", article).strip())
    out, cur = [], []
    for s in sents:
        cur.append(s)
        n = sum(len(x.split()) for x in cur)
        if n >= lo:
            if n <= hi:
                out.append(" ".join(cur))
            cur = []
        if len(out) >= take:
            break
    return out


def split_gold(rows, frac_dev=1 / 3):
    """Stratified dev/test split of one language's human test set by first label."""
    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["labels"][0]].append(r)
    for lst in by.values():
        rng.shuffle(lst)
        k = round(len(lst) * frac_dev)
        for i, r in enumerate(lst):
            r["split"] = "dev" if i < k else "test"
    return rows


# ---------------------------------------------------------------- Swahili
def swahili():
    out = []
    for r in csv.DictReader(open(IN["swa_data"], encoding="utf-8-sig")):
        if r["Split"] == "train":
            out.append(rec("swa", r["Text"], r["Domain"], "train", r["Source"], "weak:source_or_keywords", None))
        else:   # dev = passages a human reviewer kept, with the reviewer-confirmed domain
            out.append(rec("swa", r["Text"], r["Domain"], "dev", r["Source"], "gold:review_kept", None))
    for r in csv.DictReader(open(IN["swa_news_tier"], encoding="utf-8-sig")):
        out.append(rec("swa", r["Text"], r["Domain"], "train", r["Source"], "weak:news_category", None))
    # `other`: sports and entertainment news, plus national/international news chunks with no domain keyword
    news = pd.read_parquet(IN["swa_other_news"])
    quota = {"michezo": (1500, 1), "burudani": (800, 1), "kimataifa": (800, 0), "kitaifa": (1400, 0)}
    rng = random.Random(SEED)
    for cat, (n, max_hits) in quota.items():
        arts = news.loc[news.category == cat, "content"].tolist()
        rng.shuffle(arts)
        got = 0
        for a in arts:
            for c in chunks(a):
                if sw_hits(c) <= max_hits and got < n:
                    out.append(rec("swa", c, OTHER, "train", f"mollel_news:{cat}", "weak:news_category", None))
                    got += 1
            if got >= n:
                break
    # human test set: 360 passages labelled by Michael (one or two domains, or Other)
    ws = openpyxl.load_workbook(IN["swa_gold_xlsx"], read_only=True).active
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    labs = open(IN["swa_gold_labels"], encoding="utf-8").read().splitlines()
    assert len(rows) == len(labs) == 360, (len(rows), len(labs))
    gold = [rec("swa", r[2], l, None, r[1], "gold:human_label", f"swa_gold_{int(r[0])}") for r, l in zip(rows, labs)]
    return out + split_gold(gold)


# ---------------------------------------------------------------- Kinyarwanda
def kinyarwanda():
    z = zipfile.ZipFile(IN["kin_zip"])
    base = "kinyarwanda_classifier_final/data/"
    cand = pd.read_csv(io.BytesIO(z.read(base + "kinyarwanda_multidomain_candidates.csv")))
    gold = pd.read_excel(io.BytesIO(z.read(base + "Gold_Test_sample.xlsx")))
    out = [rec("kin", r.raw_text, r.domain, "train", r.source_dataset, "weak:source_default", r.id)
           for r in cand.itertuples() if isinstance(r.raw_text, str) and r.raw_text.strip()]
    g = [rec("kin", r.raw_text, r.label, None, "kin_gold_test", "gold:human_label", f"kin_gold_{r.sample_id}")
         for r in gold.itertuples() if isinstance(r.raw_text, str) and isinstance(r.label, str)]
    return out + split_gold(g)


# ---------------------------------------------------------------- Nigerian Pidgin
def pidgin():
    z = zipfile.ZipFile(IN["pcm_zip"])
    out, relab = [], 0
    for split in ("train", "dev", "test"):
        for i, line in enumerate(z.read(f"pcm_results/data/classifier/pcm_{split}.jsonl").decode("utf-8").splitlines()):
            r = json.loads(line)
            lab, src = r["label"], r.get("label_source") or "gold"
            if split == "train" and lab == "other" and src.startswith("weak") and len(set(m.lower() for m in PCM_AGRI_RX.findall(r["text"]))) >= 2:
                lab, src = "agriculture", "weak:keywords_agriculture_relabelled"; relab += 1
            out.append(rec("pcm", r["text"], lab, split, r["source"], src, f"pcm_{split}_{i}"))
    print("pcm: relabelled", relab, "keyword `other` rows as agriculture")
    return out


def extra(paths):
    out = []
    for p in paths:
        rows = [json.loads(l) for l in open(p, encoding="utf-8")] if p.endswith(".jsonl") else list(csv.DictReader(open(p, encoding="utf-8-sig")))
        for r in rows:
            out.append(rec(r["language"], r["text"], r["labels"], r["split"], r.get("source", os.path.basename(p)),
                           r.get("label_source", "gold:human_label" if r["split"] != "train" else "weak:unknown"), r.get("id")))
    return out


# ---------------------------------------------------------------- leakage control
def grams(t, n=8):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def drop_leaks(rows):
    held = [r for r in rows if r["split"] != "train"]
    exact = {re.sub(r"\W+", " ", r["text"].lower()).strip() for r in held}
    g = set().union(*(grams(r["text"]) for r in held))
    keep, dropped = [], collections.Counter()
    for r in rows:
        if r["split"] == "train":
            key = re.sub(r"\W+", " ", r["text"].lower()).strip()
            if key in exact or len(grams(r["text"]) & g) >= 3:
                dropped[r["language"]] += 1
                continue
        keep.append(r)
    return keep, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="more languages in the shared format (CSV/JSONL)")
    a = ap.parse_args()
    rows = swahili() + kinyarwanda() + pidgin() + extra(a.extra)
    rows = [r for r in rows if len(r["text"].split()) >= 3]
    # exact duplicates inside train (same language + text): keep the first
    seen, dedup = set(), []
    for r in rows:
        k = (r["language"], r["split"], r["text"].lower())
        if k in seen:
            continue
        seen.add(k); dedup.append(r)
    rows, dropped = drop_leaks(dedup)
    cnt = collections.Counter()
    for i, r in enumerate(rows):
        r["id"] = r["id"] or f"{r['language']}_{r['split']}_{i:06d}"
        cnt[(r["language"], r["split"], "|".join(r["labels"]))] += 1
    os.makedirs(OUT, exist_ok=True)
    for s in ("train", "dev", "test"):
        with open(os.path.join(OUT, f"domain_{s}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                if r["split"] == s:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"rows": len(rows), "train_rows_dropped_as_leaks": dict(dropped),
               "counts": {f"{l}/{s}/{lab}": n for (l, s, lab), n in sorted(cnt.items())}}
    json.dump(summary, open(os.path.join(OUT, "data_summary.json"), "w"), indent=1)
    tab = collections.defaultdict(collections.Counter)
    for (l, s, lab), n in cnt.items():
        tab[(l, s)]["multi" if "|" in lab else lab] += n
    for k in sorted(tab):
        print(k, dict(tab[k]))
    print("train rows dropped as dev/test leaks:", dict(dropped))


if __name__ == "__main__":
    main()

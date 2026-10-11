"""Shared-format Swahili data files for the team repo layout.
Data.csv        : Language, Domain, Source, Text, Split (train/dev), License   (licence-clear tier only)
Data_news_tier.csv : same columns, news tier (rights unresolved), kept apart
Data_row_ids.csv: row number -> record id, for traceability (not part of the shared format)
Evaluation.csv  : Language, Domain, Source, Text, Text_No_Punct, Hypothesis, Domain_Pred   (dev rows; Hypothesis and Domain_Pred are filled by the PnC model and the classifier)
Split rule: dev = passages Michael kept in a human review (gold); train = every other licence-clear passage that Michael did not mark domain_ok = NO or punctuation_ok = NO,
and that is not a near-duplicate (word 8-gram Jaccard >= 0.5) of a dev passage.
"""
import json, os, re, csv, unicodedata, collections, openpyxl
OUT = os.path.dirname(os.path.abspath(__file__)); SF = os.path.join(OUT, "shared_format"); os.makedirs(SF, exist_ok=True)
CLEAR = re.compile(r"^(CC-BY|CC-BY-SA|Legislative text excluded|CDC public-domain|Public domain \(VOA|ODC-By \(HuggingFaceFW/fineweb-2\); client approved)", re.I)
def normalise(t):
    t = unicodedata.normalize("NFC", t)
    t = t.replace("’", "'").replace("‘", "'").replace("ʼ", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\s+([,.;:?!])", r"\1", t)
key = lambda t: normalise(t).lower()
def sh(t, n=8):
    w = re.findall(r"\w+", t.lower()); return {" ".join(w[i:i+n]) for i in range(max(1, len(w)-n+1))}

recs = {}; bytext = {}
for l in open(os.path.join(OUT, "pnc_annotated.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    r["text"] = " ".join(t["pre"] + t["word"] + t["post"] for t in r["tokens"])
    r["nopunct"] = " ".join(t["word"].lower() for t in r["tokens"])
    recs[r["id"]] = r; bytext.setdefault(key(r["text"]), []).append(r["id"])

# human reviews
bad, gold = set(), set()
ws = openpyxl.load_workbook(os.path.join(OUT, "manual_validation_sample_EN.xlsx"))["Swahili"]
h = [c.value for c in ws[1]]; ix = {n: i for i, n in enumerate(h)}
for row in ws.iter_rows(min_row=2, values_only=True):
    ids = bytext.get(key(row[ix["raw_text"]]), [])
    if row[ix["domain_ok"]] == "NO" or row[ix["punctuation_ok"]] == "NO": bad.update(ids)
    if row[ix["keep"]] == "YES": gold.update(ids)
for review_file in ("fineweb_review_sample_90.xlsx", "review_health_law_60.xlsx"):
    wb = openpyxl.load_workbook(os.path.join(OUT, review_file)); rows = list(wb["Swahili"].iter_rows(min_row=2, values_only=True))
    kmap = {r[0]: r[1] for r in wb["key"].iter_rows(min_row=2, values_only=True)}
    for r in rows:
        rid = kmap[r[0]]
        if r[4] == "NO" or r[5] == "NO": bad.add(rid)
        if r[7] == "YES": gold.add(rid)
gold -= bad
gold = {g for g in gold if g in recs and CLEAR.match(recs[g]["license"] or "")}

dev_sh = [sh(recs[g]["text"]) for g in gold]
idx = collections.defaultdict(set)
for i, s in enumerate(dev_sh):
    for g in s: idx[g].add(i)
def near_dev(s8):
    c = collections.Counter(j for g in s8 for j in idx.get(g, ()))
    return bool(c) and max(c.values()) / max(1, len(s8)) >= 0.5

stats = collections.Counter(); drop = collections.Counter(); out_rows = []; news_rows = []; ids = []; seen_text = set()
cols = ["Language", "Domain", "Source", "Text", "Split", "License"]
for rid, r in recs.items():
    row = ["swa", r["domain"], r["source_dataset"], r["text"], None, r["license"]]
    if r["set"] == "news_tier": row[4] = "train"; news_rows.append((rid, row)); continue
    if not CLEAR.match(r["license"] or ""): drop["not_licence_clear"] += 1; continue
    if key(r["text"]) in seen_text: drop["exact_duplicate_text"] += 1; continue
    seen_text.add(key(r["text"]))
    if rid in gold: row[4] = "dev"
    elif rid in bad: drop["marked_domain_or_punctuation_NO"] += 1; continue
    elif near_dev(sh(r["text"])): drop["near_duplicate_of_dev"] += 1; continue
    else: row[4] = "train"
    out_rows.append((rid, row)); stats[(r["domain"], row[4])] += len(r["tokens"])
with open(os.path.join(SF, "Data.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(cols); [w.writerow(r) for _, r in out_rows]
with open(os.path.join(SF, "Data_news_tier.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(cols); [w.writerow(r) for _, r in news_rows]
with open(os.path.join(SF, "Data_row_ids.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "row", "id"]); [w.writerow(["Data.csv", i + 2, rid]) for i, (rid, _) in enumerate(out_rows)]; [w.writerow(["Data_news_tier.csv", i + 2, rid]) for i, (rid, _) in enumerate(news_rows)]
with open(os.path.join(SF, "Evaluation.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["Language", "Domain", "Source", "Text", "Text_No_Punct", "Hypothesis", "Domain_Pred"])
    for rid, row in out_rows:
        if row[4] == "dev": w.writerow(["swa", row[1], row[2], row[3], recs[rid]["nopunct"], "", ""])
D = ["agriculture", "finance", "health", "law", "telecoms"]
print(f"{'domain':12s} {'train words':>12s} {'dev words':>10s}")
for d in D: print(f"{d:12s} {stats[(d,'train')]:12,d} {stats[(d,'dev')]:10,d}")
print("totals train", sum(v for (d, s), v in stats.items() if s == "train"), "dev", sum(v for (d, s), v in stats.items() if s == "dev"))
print("dev passages", len(gold), "dropped", dict(drop), "news-tier rows", len(news_rows), "bad ids", len(bad))
json.dump({"words": {f"{d}/{s}": v for (d, s), v in stats.items()}, "dev_passages": len(gold), "dropped": dict(drop), "news_tier_rows": len(news_rows)}, open(os.path.join(SF, "split_stats.json"), "w"), indent=1)

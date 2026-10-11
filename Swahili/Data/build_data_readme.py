"""Write shared_format/README.md (statistics only) from Data.csv and Data_news_tier.csv."""
import csv, collections, json, os
SF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_format")
rows = list(csv.DictReader(open(os.path.join(SF, "Data.csv"), encoding="utf-8-sig"))); news = list(csv.DictReader(open(os.path.join(SF, "Data_news_tier.csv"), encoding="utf-8-sig")))
ver = json.load(open(os.path.join(os.path.dirname(SF), "human_verified_words.json")))
D = ["agriculture", "finance", "health", "law", "telecoms"]
def fam(r):
    s = r["Source"].lower(); l = r["License"]
    if "fineweb" in s: return "FineWeb-2 (ODC-By, client approved)"
    if "wikipedia" in s or l.startswith("CC-BY-SA-4.0 (Wikipedia") or "GFDL" in l or s in ("kilimo", "mahindi", "umwagiliaji"): return "Wikipedia (CC BY-SA)"
    if "voa" in s: return "VOA Swahili (public domain)"
    if "tanzlii" in s: return "TanzLII (CC BY, to confirm)"
    return "Documents with a CC licence (TARI, IITA, guides, others)"
W = collections.defaultdict(collections.Counter); P = collections.Counter(); S = collections.Counter()
for r in rows:
    n = len(r["Text"].split()); W[r["Domain"]][fam(r)] += n; P[(r["Domain"], r["Split"])] += 1; S[(r["Domain"], r["Split"])] += n
fams = sorted({f for d in W for f in W[d]}); ndev = sum(v for (d, s), v in P.items() if s == "dev")
o = ["# Swahili data statistics (words, whitespace-separated)", "", "Files: `Data.csv` (licence-clear tier, columns `Language, Domain, Source, Text, Split, License`), `Evaluation.csv` (dev rows with `Text_No_Punct`, `Hypothesis`, `Domain_Pred`), `Data_news_tier.csv` (news, rights unresolved, kept apart). Language code: `swa`. Corpus text is not in the public repository; see the Swahili README for why.", "",
     "## Words per domain and split", "", "| Domain | Train words | Dev words | Train passages | Dev passages | Human-verified words (all reviews) |", "|---|---:|---:|---:|---:|---:|"]
for d in D: o.append(f"| {d} | {S[(d,'train')]:,} | {S[(d,'dev')]:,} | {P[(d,'train')]:,} | {P[(d,'dev')]:,} | {ver['human_verified_words_by_domain'].get(d, 0):,} |")
o.append(f"| **Total** | **{sum(v for (d,s),v in S.items() if s=='train'):,}** | **{sum(v for (d,s),v in S.items() if s=='dev'):,}** | {sum(v for (d,s),v in P.items() if s=='train'):,} | {ndev} | **{ver['total']:,}** |")
o += ["", f"Dev = passages a human reviewer kept ({ndev} passages). The dev split is small; more reviewed passages will be added. Human-verified words include reviewed passages that are not in the annotated set.", "",
      "## Words per domain by source family (train and dev)", "", "| Source family | " + " | ".join(D) + " |", "|---|" + "---:|" * len(D)]
for f in fams: o.append(f"| {f} | " + " | ".join(f"{W[d][f]:,}" for d in D) + " |")
nw = collections.Counter()
for r in news: nw[r["Domain"]] += len(r["Text"].split())
o += ["", "## News tier (separate file, not counted above)", "", "| Domain | Words |", "|---|---:|"] + [f"| {d} | {nw[d]:,} |" for d in D if nw[d]]
o += ["", "Dataset cards say CC BY 4.0, but newspaper rights are not cleared; glued sentences in the source were repaired mechanically.", "", "## How the split was made", "",
      "1. Dev: passages kept in a human review. 2. Train: every other licence-clear passage, except those a reviewer marked wrong domain or unusable punctuation and near-duplicates (word 8-gram Jaccard 0.5 or more) of a dev passage. 3. Exact duplicate texts were dropped. Script: `build_shared_format.py`.", "",
      "## Caveats", "", "- Passing the rules is not human verification; about 55% of FineWeb-2 passages, 60% of health and 43% of law passages (Wikipedia and TanzLII) were kept in review.", "- FineWeb-2 use was approved verbally by the client; written confirmation is pending.",
      "- TanzLII passages carry a CC BY licence recorded from the earlier source register; the PDFs print none and the site's licence note is still to be confirmed. In review, 4 of 13 TanzLII passages were kept; statute fragments often start or stop mid-section."]
open(os.path.join(SF, "README.md"), "w", encoding="utf-8").write("\n".join(o) + "\n"); print("\n".join(o[4:14]))

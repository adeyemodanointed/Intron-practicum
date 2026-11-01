"""Build one train/dev/test set for the multilingual domain classifier.

Record (JSONL): id, language, text, labels (list: one or two domains, or ["other"]), split, source, label_source.
label_source: gold:* = a person or a human dataset label assigned it; weak:* = rule, keyword or source default.

Languages: swa, kin, pcm, yor (Yoruba: synthetic, see yoruba()). More can be added with --extra (CSV or JSONL with
columns language, text, labels, split; labels like "finance" / "finance|law" / "other").

Splits
  train  every weak/silver row, plus gold training rows (pcm and swa/yor MasakhaNEWS topic labels mapped to our domains)
  dev    one third of each language's human test set (stratified, seed 13) + swa human-kept review passages;
         used only to tune the per-domain thresholds
  test   the other two thirds of each human test set (pcm: its own human test split), plus the MasakhaNEWS Yoruba
         test split (real Yoruba news, health vs other only)
Cross-validation: every dev/test row also gets a `fold` (0-4, stratified by language, label kind and first label) and
is written to domain_eval.jsonl. For fold k: test = fold k, dev = fold (k+1) % 5, the other three folds are training data.
External topic datasets (train only; --no-noncommercial leaves them out):
  MasakhaNEWS swa/yor  business -> finance, health -> health, sports/entertainment/religion/politics -> other
                       (politics articles with law words dropped), technology dropped. Licence unclear (AFL-3.0 in the
                       metadata, CC BY-NC 4.0 in the card). Our existing Pidgin gold data is MasakhaNEWS pcm.
  KINNEWS (kin)        health -> health; politics, sport, entertainment, history, tourism, culture, fashion, religion,
                       education, relationship -> other (politics with law words dropped); economy, technology,
                       environment dropped. MIT licence; article copyright stays with the news sources.
Leakage: a training row is dropped if it matches a dev/test text exactly or shares 3+ word 8-grams with one.

Usage: python build_domain_data.py [--extra more.csv ...] [--no-noncommercial] [--out data]
"""
import argparse, collections, csv, glob, io, json, os, random, re, zipfile, zlib

import openpyxl
import pandas as pd

from domain_model import DOMAINS, OTHER, fold, norm_labels

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
    "yor_synthetic": os.path.join(HERE, "inputs", "yoruba_synthetic_clean.csv.gz"),
    "masakhanews": os.path.join(HERE, "inputs", "external", "masakhanews"),   # {lang}/{train,dev,test}.tsv
    "kinnews_zip": os.path.join(HERE, "inputs", "external", "kinnews", "KINNEWS.zip"),
    # v4 additions
    "du": os.path.join(HERE, "inputs", "external", "digitalumuganda"),        # DigitalUmuganda health text (Kinyarwanda)
    "sib200": os.path.join(HERE, "inputs", "external", "sib200"),             # {kin,swh}_Latn/{train,dev,test}.tsv
    "swa_shared": os.path.join(ROOT, "outputs", "deliverables-2026-10-04", "swahili_shared.jsonl"),   # incl. Afrivoice Swahili
    "afrivoice_kin": os.path.join(HERE, "inputs", "external", "afrivoice_kin"),  # gated manifests (optional)
}
OUT = os.path.join(HERE, "data")
SEED = 13
N_FOLDS = 5

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


# Law words per language (diacritics folded). Politics articles that mention the law are not used as `other`,
# because our annotators label court, legislation and police texts as law.
LAW_WORDS = {
    "swa": SW_LEX["law"],
    "kin": ("urukiko inkiko itegeko amategeko ubutabera umucamanza abacamanza icyaha ibyaha ubushinjacyaha uregwa "
            "abaregwa gereza igifungo gufungwa avoka umunyamategeko polisi").split(),
    "yor": "ile-ejo ileejo adajo ofin olopaa agbejoro ewon idajo".split(),
}
LAW_RX = {l: re.compile(r"\b(" + "|".join(map(re.escape, v)) + r")\w*", re.I) for l, v in LAW_WORDS.items()}
MASAKHANEWS_MAP = {"business": "finance", "health": "health", "sports": OTHER, "entertainment": OTHER,
                   "religion": OTHER, "politics": OTHER}   # technology: dropped (not only telecoms)
KINNEWS_MAP = {1: OTHER, 2: OTHER, 4: "health", 5: OTHER, 6: OTHER, 8: OTHER, 9: OTHER, 10: OTHER, 11: OTHER,
               13: OTHER, 14: OTHER}                       # 3 economy, 7 technology, 12 environment: dropped
KINNEWS_POLITICS = 1
# v4: KINNEWS categories dropped above, kept when the article has enough words of one domain (diacritics folded)
KIN_LEX = {
    "finance": ("amafaranga banki inguzanyo imisoro umusoro ishoramari kuzigama ubwizigame ubukungu isoko ibiciro inyungu "
                "konti imari amadovize frw").split(),
    "telecoms": ("telefoni telefone itumanaho interineti murandasi mtn airtel tigo momo simukadi ikoranabuhanga mudasobwa "
                 "radiyo televiziyo network").split(),
    "agriculture": ("ubuhinzi abahinzi umuhinzi ubworozi aborozi ifumbire imbuto umusaruro amatungo inka ibihingwa igihingwa "
                    "imyumbati ibigori ikawa icyayi imirima umurima").split(),
    "health": ("indwara ibitaro muganga abaganga imiti urukingo inkingo abarwayi umurwayi ivuriro mituweli imirire malariya "
               "virusi ubuvuzi kwivuza").split(),
}
KIN_RX = {d: re.compile(r"\b(" + "|".join(map(re.escape, v)) + r")\w*", re.I) for d, v in KIN_LEX.items()}
KINNEWS_EXTRA = {3: ("finance", 3), 7: ("telecoms", 2), 12: ("agriculture", 3)}   # label: (domain, distinct words needed)
# Yoruba domain words (diacritics folded), for keyword labels on real Yoruba news / Wikipedia text (WURA)
YOR_LEX = {
    "agriculture": "ogbin agbe irugbin ajile ikore maalu darandaran agbado gbaguda iresi oko-ogbin osin-eran".split(),
    "finance": "ile-ifowopamo ileifowopamo banki naira owo-ori idokowo oro-aje gbese awin owo-osu dola ileese-owo".split(),
    "health": "ilera aisan arun ile-iwosan ileewosan dokita oogun ajesara korona alaisan noosi".split(),
    "law": "ile-ejo ileejo adajo ofin olopaa agbofinro agbejoro loya ewon esun idajo ijiya".split(),
    "telecoms": "foonu ero-ibanisoro ibanisoro ayelujara intaneeti inteneeti mtn glo airtel 9mobile ncc sim redio telifisan".split(),
}
YOR_RX = {d: re.compile(r"\b(" + "|".join(map(re.escape, v)) + r")\w*", re.I) for d, v in YOR_LEX.items()}
WURA_KEEP = ("yo.wikipedia.org", "alaroye.org", "www.bbc.com", "www.awikonko.com.ng", "yoruba.von.gov.ng", "www.asejere.net",
             "www.theyorubablog.com")   # human-written news / encyclopedia; machine-translated sites are left out
SIB_MAP = {"health": "health", "politics": OTHER, "sports": OTHER, "entertainment": OTHER, "travel": OTHER, "geography": OTHER}
# Pidgin domain words (English-based spelling plus common Pidgin forms), for keyword labels on FineWeb-2 Pidgin
PCM_LEX = {
    "agriculture": PCM_AGRI,
    "finance": ("money moni bank banks naira dollar dollars loan loans tax taxes economy business market price prices "
                "invest investment investors salary budget profit inflation cbn kobo payment crypto bitcoin stock shares "
                "fund funds debt revenue").split(),
    "health": ("health hospital hospitals doctor doctors nurse nurses medicine medicines drug drugs disease diseases sick "
               "sickness vaccine vaccines malaria covid coronavirus virus patient patients treatment cancer hiv pregnant "
               "pregnancy clinic ncdc ebola infection").split(),
    "law": ("court courts judge judges law laws lawyer lawyers police arrest arrested prison jail crime crimes criminal "
            "justice sue suit efcc constitution legal convicted sentence sentenced tribunal").split(),
    "telecoms": ("phone phones network internet sim mtn glo airtel 9mobile ncc sms telecom telecoms telecommunication "
                 "broadband radio television tv 5g 4g mobile").split(),
}
PCM_RX = {d: re.compile(r"\b(" + "|".join(map(re.escape, v)) + r")\b", re.I) for d, v in PCM_LEX.items()}
# FineWeb-2 (ODC-By) sites: Pidgin keeps every site except spam, joke and song-lyric ones; Yoruba keeps only human-written
# news, government, encyclopedia and religious sites (most other Yoruba hosts are machine-translated blogs)
FW2_PCM_SKIP = ("www.freenaijalyrics.com", "www.lolcatbible.com", "ebony.vaginasandlips.com", "pronunciaciones.com",
                "aderconstanta.info", "sok-kosovo.org", "hrmetrics.info", "en.skuboplay.com", "www.mobofree.com", "global-asp.github.io")
FW2_YOR_KEEP = ("alaroye.org", "alaroye.tv", "www.bbc.com", "iroyinowuro.com.ng", "www.jw.org", "www.biblica.com",
                "iweirohinapere.com", "www.iweirohinapere.com", "yoruba.von.gov.ng", "radionigeriaibadan.gov.ng",
                "yo.wikipedia.org", "ooduarere.com", "asa.ooduarere.com", "esinislam.com", "yo.globalvoices.org", "orisun.tv",
                "islamhouse.com")
# Scripture and preaching: `other` whatever words they use (our annotators label Bible verses about courts or seeds `other`)
FW2_RELIGIOUS = ("pidginbible.com", "www.bible.is", "cofwparks.org", "www.jw.org", "www.biblica.com", "esinislam.com", "islamhouse.com")


def kin_hits(domain, t):
    return len({m.lower() for m in KIN_RX[domain].findall(fold(t))})


def law_hits(lang, t):
    return len({m.lower() for m in LAW_RX[lang].findall(fold(t))})


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


# ---------------------------------------------------------------- Yoruba
def yoruba():
    """Gemini-generated Yoruba sentences, 2,000 per domain, labels unreviewed; no `other` class. Not used by default:
    the client asked for no LLM-generated Yoruba in training (--yor-synthetic brings it back for the old comparisons).
    Its own split is kept (train / validation -> dev / test), but every row is weak:synthetic, so the Yoruba
    test score is reported separately and left out of the human-labelled ALL score."""
    d = pd.read_csv(IN["yor_synthetic"])
    sp = {"train": "train", "validation": "dev", "test": "test"}
    return [rec("yor", r.Text, r.Domain, sp[r.Split], f"synthetic:{r.Generator_Model}", "weak:synthetic_unreviewed", r.ID)
            for r in d.itertuples() if isinstance(r.Text, str)]


# ---------------------------------------------------------------- external topic datasets
def masakhanews(lang):
    """MasakhaNEWS train/dev/test -> training rows (whole-sentence chunks); the Yoruba test split -> test rows."""
    out, skipped = [], collections.Counter()
    for split in ("train", "dev", "test"):
        d = pd.read_csv(os.path.join(IN["masakhanews"], lang, f"{split}.tsv"), sep="\t")
        held = lang == "yor" and split == "test"
        for i, r in enumerate(d.itertuples()):
            lab = MASAKHANEWS_MAP.get(r.category)
            if lab is None or not isinstance(r.text, str):
                skipped[r.category] += 1
                continue
            if r.category == "politics" and law_hits(lang, r.text) >= 2:
                skipped["politics_with_law_words"] += 1
                continue
            parts = chunks(r.text, take=1 if held else 2) or [" ".join(r.text.split()[:80])]
            for j, c in enumerate(parts):
                out.append(rec(lang, c, lab, "test" if held else "train", f"masakhanews:{lang}:{split}:{r.category}",
                               "gold:dataset_label", f"mnews_{lang}_{split}_{i}_{j}"))
    print(f"masakhanews {lang}: {len(out)} rows, skipped {dict(skipped)}")
    return out


def kinnews(v4=False):
    """KINNEWS (raw) train+test -> Kinyarwanda training rows, one chunk per article; gives Kinyarwanda its `other` text.
    v4: also economy/technology/environment articles with enough finance/telecoms/agriculture words, and politics
    articles with 4+ law words as law."""
    csv.field_size_limit(10 ** 9)
    z = zipfile.ZipFile(IN["kinnews_zip"])
    out, skipped = [], collections.Counter()
    for split in ("train", "test"):
        rows = list(csv.reader(io.TextIOWrapper(z.open(f"KINNEWS/raw/{split}.csv"), encoding="utf-8", errors="replace")))
        for i, r in enumerate(rows[1:]):
            label, en, content = int(r[0]), r[2], r[5]
            lab = KINNEWS_MAP.get(label)
            if lab is None and v4 and label in KINNEWS_EXTRA:
                dom, need = KINNEWS_EXTRA[label]
                lab = dom if kin_hits(dom, content) >= need else None
            if lab is None:
                skipped[en] += 1
                continue
            if label == KINNEWS_POLITICS and law_hits("kin", content) >= 2:
                if v4 and law_hits("kin", content) >= 4:   # clearly about courts / legislation
                    lab = "law"
                else:
                    skipped["politics_with_law_words"] += 1
                    continue
            for c in chunks(content, take=1):
                out.append(rec("kin", c, lab, "train", f"kinnews:{en}", "weak:news_category", f"kinnews_{split}_{i}"))
    print(f"kinnews: {len(out)} rows, skipped {dict(skipped)}")
    return out


# ---------------------------------------------------------------- v4 additions (data the earlier versions did not use)
def kin_health(noncommercial=True, cap=8000, llm_text=False):
    """Kinyarwanda health text from DigitalUmuganda: RBC community-health material (CC-BY-2.0, chunked in document order)
    and, unless commercial-safe, validated health sentence translations (NMT_Health, licence not stated; 10+ words).
    The translations whose English source was written by GPT-4 are left out unless llm_text (no LLM-generated text)."""
    out = []
    lines = [l.strip() for l in open(os.path.join(IN["du"], "Monolingual_health_dataset", "rbc_kinyarwanda_health_dataset.txt"),
                                     encoding="utf-8", errors="replace") if l.strip()]
    cur, n = [], 0
    for l in lines:   # 25-60 word passages from consecutive lines
        cur.append(l)
        w = sum(len(x.split()) for x in cur)
        if w >= 25:
            if w <= 60:
                out.append(rec("kin", " ".join(cur), "health", "train", "digitalumuganda:rbc_health", "weak:source_default", f"rbc_{n}"))
                n += 1
            cur = []
    if noncommercial:
        rows = list(csv.DictReader(open(os.path.join(IN["du"], "NMT_Health_parallel_data_en_kin", "health_train_data.tsv"),
                                        encoding="utf-8", errors="replace"), delimiter="\t"))
        rows = [r for r in rows if r["validation_state"] == "1" and len(r["kin"].split()) >= 10 and (llm_text or r["domain"] != "gpt-4")]
        random.Random(SEED).shuffle(rows)
        out += [rec("kin", r["kin"], "health", "train", f"digitalumuganda:nmt_health:{r['domain']}", "weak:source_default", f"nmth_{r['id']}")
                for r in rows[:cap]]
    print(f"kin health (DigitalUmuganda): {len(out)} rows")
    return out


def sib200():
    """SIB-200 (CC BY-SA 4.0) Kinyarwanda, Swahili and Yoruba sentences: health -> health; politics, sports, entertainment,
    travel, geography -> other; science/technology dropped."""
    out = []
    for code, lang in (("kin_Latn", "kin"), ("swh_Latn", "swa"), ("yor_Latn", "yor")):
        for split in ("train", "dev", "test"):
            d = pd.read_csv(os.path.join(IN["sib200"], code, f"{split}.tsv"), sep="\t")
            out += [rec(lang, r.text, SIB_MAP[r.category], "train", f"sib200:{r.category}", "gold:dataset_label", f"sib200_{lang}_{r.index_id}")
                    for r in d.itertuples() if r.category in SIB_MAP]
    print(f"sib200: {len(out)} rows")
    return out


def swa_shared():
    """Swahili shared-format passages (AI-screened), incl. ~4,500 Afrivoice Swahili texts not in Data.csv."""
    out = []
    for l in open(IN["swa_shared"], encoding="utf-8"):
        r = json.loads(l)
        out.append(rec("swa", r["raw_text"], r["domain"], "train", r["source_dataset"], "weak:ai_screened", None))
    print(f"swahili shared: {len(out)} rows")
    return out


def pcm_review():
    """60 Pidgin texts a person checked (pcm_annotation_filled.xlsx): the only human Pidgin labels for law and telecoms.
    Added to the evaluation pool."""
    z = zipfile.ZipFile(IN["pcm_zip"])
    d = pd.read_excel(io.BytesIO(z.read("pcm_results/reviews/pcm_annotation_filled.xlsx")))
    out = []
    for r in d.itertuples():
        lab = r.predicted if str(r.correct).strip().upper() == "Y" or not isinstance(r.true_domain, str) else r.true_domain
        out.append(rec("pcm", r.text, lab, "test", f"pcm_review:{r.source}", "gold:human_review", f"pcm_review_{r.id}"))
    print(f"pcm human review: {len(out)} rows")
    return out


def afrivoice_kin():
    """Afrivoice Kinyarwanda transcripts the earlier data did not use (gated, CC-BY-4.0; photo descriptions, the style of
    the human Kinyarwanda test texts), from inputs/external/afrivoice_kin/afrivoice_kin_unused_transcripts.jsonl
    (55,471 rows sampled in the browser: all agriculture/financial/health validation+test, 25,000 government,
    20,000 education + scripted education):
      agriculture/financial/health val+test -> that domain (their train splits are in kinyarwanda_multidomain_candidates.csv)
      government -> law with 2+ law words, else other unless it has 2+ words of another domain (then left out)
      education, scripted education -> other, left out when it has 2+ words of any domain (or of the law list)"""
    p = os.path.join(IN["afrivoice_kin"], "afrivoice_kin_unused_transcripts.jsonl")
    if not os.path.exists(p):
        return []
    dom = {"agriculture": "agriculture", "financial": "finance", "health": "health"}
    out, skipped = [], collections.Counter()
    for i, l in enumerate(open(p, encoding="utf-8")):
        r = json.loads(l)
        cat, split = r["config"].rsplit("_", 1)
        t = r["raw_text"]
        if cat in dom:
            lab = dom[cat]
        else:
            law = law_hits("kin", t)
            other_dom = max(kin_hits(d, t) for d in KIN_RX)
            if cat == "government" and law >= 2:
                lab = "law"
            elif other_dom >= 2 or law >= 2:
                skipped[f"{cat}_with_domain_words"] += 1
                continue
            else:
                lab = OTHER
        out.append(rec("kin", t, lab, "train", f"afrivoice_kin:{cat}:{split}", "weak:source_default", f"afv_{r['config']}_{i}"))
    print(f"afrivoice kin (unused parts): {len(out)} rows, skipped {dict(skipped)}")
    return out


def wura_yor(cap=3000, need=3):
    """Real Yoruba text from WURA (Apache-2.0; human-written news and Wikipedia only), 30-80 word chunks labelled by
    Yoruba domain words: a domain when the chunk has `need`+ distinct words of it and at most one of any other domain.
    At most `cap` chunks per domain, seed 13."""
    by = collections.defaultdict(list)
    for split in ("train", "eval"):
        for i, l in enumerate(open(os.path.join(HERE, "inputs", "external", "wura_yor", f"yor_{split}.jsonl"), encoding="utf-8")):
            r = json.loads(l)
            host = re.sub(r"^https?://", "", r.get("url") or "").split("/")[0]
            if host not in WURA_KEEP:
                continue
            for j, c in enumerate(chunks(r.get("content") or "", take=3)):
                f = fold(c)
                hits = {d: len({m.lower() for m in rx.findall(f)}) for d, rx in YOR_RX.items()}
                best = max(hits, key=hits.get)
                if hits[best] >= need and all(v <= 1 for d, v in hits.items() if d != best):
                    by[best].append(rec("yor", c, best, "train", f"wura:{host}", "weak:keywords", f"wura_{split}_{i}_{j}"))
    rng, out = random.Random(SEED), []
    for d, lst in sorted(by.items()):
        rng.shuffle(lst)
        out += lst[:cap]
    print("wura yor (keyword labels):", {d: f"{min(len(v), cap)} of {len(v)}" for d, v in sorted(by.items())})
    return out


def fineweb2(held, cap=3000, need=3):
    """Real Pidgin and Yoruba web text from FineWeb-2 (ODC-By; inputs/external/fineweb2/<lang>_Latn/{train,test}), the
    source of many of the team-validated texts. 20-70 word chunks (up to 3 per page), labelled:
      religious sites (FW2_RELIGIOUS) -> other
      otherwise a domain when the chunk has `need`+ distinct words of it and at most one of any other domain
      (same rule as wura_yor), `other` when it has no domain word at all, else left out.
    At most `cap` chunks per language and label, seed 13. Chunks containing (or contained in) a held-out text are left out."""
    import pyarrow.parquet as pq
    norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
    lex = {"pcm": PCM_RX, "yor": YOR_RX}
    out = []
    for lang, code in (("pcm", "pcm_Latn"), ("yor", "yor_Latn")):
        held_l = [norm(t) for t in held.get(lang, []) if len(t.split()) >= 6]
        by = collections.defaultdict(list)
        for split in ("train", "test"):
            p = os.path.join(HERE, "inputs", "external", "fineweb2", code, split, "000_00000.parquet")
            if not os.path.exists(p):
                continue
            t = pq.read_table(p, columns=["url", "text"]).to_pydict()
            for i, (url, doc) in enumerate(zip(t["url"], t["text"])):
                host = re.sub(r"^https?://", "", url or "").split("/")[0]
                if (lang == "pcm" and host in FW2_PCM_SKIP) or (lang == "yor" and host not in FW2_YOR_KEEP):
                    continue
                for j, c in enumerate(chunks(doc, lo=20, hi=70, take=3)):
                    n = norm(c)
                    if any(h in n or n in h for h in held_l):
                        continue
                    f = fold(c) if lang == "yor" else c
                    hits = {d: len({m.lower() for m in rx.findall(f)}) for d, rx in lex[lang].items()}
                    best = max(hits, key=hits.get)
                    if host in FW2_RELIGIOUS:
                        lab, ls = OTHER, "weak:source_default"
                    elif hits[best] >= need and all(v <= 1 for d, v in hits.items() if d != best):
                        lab, ls = best, "weak:keywords"
                    elif not any(hits.values()):
                        lab, ls = OTHER, "weak:keywords"
                    else:
                        continue
                    by[lab].append(rec(lang, c, lab, "train", f"fineweb2:{host}", ls, f"fw2_{code}_{split}_{i}_{j}"))
        rng = random.Random(SEED)
        for d, lst in sorted(by.items()):
            rng.shuffle(lst)
            out += lst[:cap]
        print(f"fineweb-2 {lang}:", {d: f"{min(len(v), cap)} of {len(v)}" for d, v in sorted(by.items())})
    return out


def team_validated():
    """Texts a person labelled in the team's Practicum sheet (Manual Validation: Kinyarwanda, Swahili, Pidgin; the Yoruba
    tab): all six labels, from WURA, FineWeb-2, Wikipedia, XL-Sum, MAFAND, MasakhaNEWS and the Gazette. They join the
    cross-validation pool (own strata); main() skips those already in it (e.g. the Swahili ones are our Swahili set)."""
    p = os.path.join(HERE, "inputs", "external", "team_sheet", "team_sheet_labels.jsonl")
    out = []
    for r in map(json.loads, open(p, encoding="utf-8")):
        if r["tab"] in ("manual_validation", "sheet11_yoruba") and r["label"].strip():
            out.append(rec(r["language"], r["text"], r["label"], "test", f"team_sheet:{r['source']}", "gold:team_manual_validation",
                           f"team_{r['tab']}_{r['row']}"))
    print(f"team manually validated: {len(out)} rows")
    return out


def kin_telecom_synthetic():
    """2,000 Gemini-written Kinyarwanda telecoms sentences from the team sheet (tab kinyarwanda_telecom_synthetic_seed:
    mobile internet, SMS/USSD, calls, coverage, ...); unreviewed, so weak labels. Kinyarwanda telecoms has few texts."""
    p = os.path.join(HERE, "inputs", "external", "team_sheet", "team_sheet_labels.jsonl")
    if not os.path.exists(p):
        return []
    out = [rec("kin", r["text"], r["label"], "train", f"team_sheet:synthetic:{r.get('generator', '')}", "weak:synthetic_unreviewed",
               f"kin_telsyn_{r['row']}")
           for r in map(json.loads, open(p, encoding="utf-8")) if r["tab"] == "kin_telecom_synthetic_seed"]
    print(f"kin telecoms synthetic (team sheet): {len(out)} rows")
    return out


def assign_folds(rows):
    """Stratified 5-fold assignment of every dev/test row (by language, label kind and first label). Each stratum is
    shuffled with its own seed, so a row gets the same fold in every data variant (e.g. with --no-noncommercial)."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["split"] != "train":   # rows added later get their own strata, so earlier rows keep their folds
            kind = r["label_source"].split(":")[0] + ("_review" if r["label_source"].endswith("human_review") else "")                 + ("_team" if r["label_source"].endswith("team_manual_validation") else "")
            by[(r["language"], kind, r["labels"][0])].append(r)
    for k, lst in by.items():
        key = "/".join(k)
        random.Random(f"{SEED}/{key}").shuffle(lst)
        start = zlib.crc32(key.encode()) % N_FOLDS   # small strata start on different folds
        for i, r in enumerate(lst):
            r["fold"] = (start + i) % N_FOLDS


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


def drop_leaks(rows, extra_held=()):
    """extra_held: texts of a separate test set (e.g. data_teamtest) that training must not contain either."""
    if extra_held:   # the separate test set is kept out of the cross-validation pool too (the final model trains on it)
        ex_keys = {re.sub(r"\W+", " ", t.lower()).strip() for t in extra_held}
        ex_g = set().union(*(grams(t) for t in extra_held))
        n0 = len(rows)
        rows = [r for r in rows if r["split"] == "train" or not (re.sub(r"\W+", " ", r["text"].lower()).strip() in ex_keys
                                                                 or len(grams(r["text"]) & ex_g) >= 3)]
        print(f"held-out test texts removed from the evaluation pool: {n0 - len(rows)}")
    held = [r for r in rows if r["split"] != "train"] + [{"text": t} for t in extra_held]
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
    ap.add_argument("--no-noncommercial", action="store_true", help="leave out MasakhaNEWS swa/yor and KINNEWS (licences)")
    ap.add_argument("--v4", action="store_true", help="add the v4 sources: DigitalUmuganda Kinyarwanda health text, SIB-200, "
                                                      "Swahili shared passages, the Pidgin review sheet, more KINNEWS "
                                                      "categories and (if downloaded) the gated Afrivoice Kinyarwanda parts")
    ap.add_argument("--yor-synthetic", action="store_true", help="include the Gemini-generated Yoruba set (off by default: "
                                                                  "the client asked for no LLM-generated training text)")
    ap.add_argument("--team-validated", action="store_true", help="--v4: add the team's manually validated texts to the "
                                                                   "cross-validation pool (they then train other folds)")
    ap.add_argument("--fineweb2", action="store_true", help="--v4: add FineWeb-2 Pidgin and Yoruba web text (religious sites "
                                                             "as other, the rest labelled by domain words)")
    ap.add_argument("--llm-text", action="store_true", help="--v4: also the other LLM-generated texts (Gemini Kinyarwanda "
                                                             "telecoms sentences, health translations of GPT-4 English); off by default")
    ap.add_argument("--relabel", nargs="*", default=[], help="JSONL with language, text, labels, label_source (e.g. LLM "
                                                               "multi-domain labels): training rows with the same text get them")
    ap.add_argument("--heldout", nargs="*", default=[], help="JSONL test sets kept out of training (leakage filter only, "
                                                              "e.g. data_teamtest/domain_teamtest.jsonl)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    rows = swahili() + kinyarwanda() + pidgin() + (yoruba() if a.yor_synthetic else []) + extra(a.extra)
    if not a.no_noncommercial:
        rows += masakhanews("swa") + masakhanews("yor") + kinnews(v4=a.v4)
    if a.v4:
        rows += kin_health(noncommercial=not a.no_noncommercial, llm_text=a.llm_text) + sib200() + swa_shared() + pcm_review()
        rows += afrivoice_kin() + wura_yor() + (kin_telecom_synthetic() if a.llm_text else [])
        if a.team_validated:
            pool = [r for r in rows if r["split"] != "train"]
            pk, pg = {re.sub(r"\W+", " ", r["text"].lower()).strip() for r in pool}, set().union(*(grams(r["text"]) for r in pool))
            team = [r for r in team_validated() if not (re.sub(r"\W+", " ", r["text"].lower()).strip() in pk or len(grams(r["text"]) & pg) >= 3)]
            print(f"team manually validated added to the evaluation pool: {len(team)} (the rest are already in it)")
            rows += team
        if a.fineweb2:
            held = collections.defaultdict(list)
            for r in rows:
                if r["split"] != "train":
                    held[r["language"]].append(r["text"])
            rows += fineweb2(held)
    if a.relabel:
        key = lambda l, t: (l, re.sub(r"\W+", " ", str(t).lower()).strip())
        new = {key(r["language"], r["text"]): r for p in a.relabel for r in map(json.loads, open(p, encoding="utf-8"))}
        n = 0
        for r in rows:
            m = new.get(key(r["language"], r["text"])) if r["split"] == "train" else None
            if m:
                r["labels"], r["label_source"] = norm_labels(m["labels"]), m["label_source"]; n += 1
        print(f"relabelled {n} training rows from {a.relabel}")
    rows = [r for r in rows if len(r["text"].split()) >= 3]
    # exact duplicates inside train (same language + text): keep the first
    seen, dedup = set(), []
    for r in rows:
        k = (r["language"], r["split"], r["text"].lower())
        if k in seen:
            continue
        seen.add(k); dedup.append(r)
    held_texts = [json.loads(l)["text"] for p in a.heldout for l in open(p, encoding="utf-8")]
    rows, dropped = drop_leaks(dedup, held_texts)
    cnt = collections.Counter()
    for i, r in enumerate(rows):
        r["id"] = r["id"] or f"{r['language']}_{r['split']}_{i:06d}"
        cnt[(r["language"], r["split"], "|".join(r["labels"]))] += 1
    assign_folds(rows)
    os.makedirs(a.out, exist_ok=True)
    for s in ("train", "dev", "test", "eval"):
        with open(os.path.join(a.out, f"domain_{s}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                if r["split"] == s or (s == "eval" and r["split"] != "train"):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    folds = collections.Counter((r["language"], r["fold"]) for r in rows if r["split"] != "train")
    summary = {"rows": len(rows), "noncommercial_sources": not a.no_noncommercial,
               "train_rows_dropped_as_leaks": dict(dropped),
               "eval_rows_per_fold": {f"{l}/{k}": n for (l, k), n in sorted(folds.items())},
               "train_sources": dict(collections.Counter(r["source"].split(":")[0] for r in rows if r["split"] == "train")),
               "counts": {f"{l}/{s}/{lab}": n for (l, s, lab), n in sorted(cnt.items())}}
    json.dump(summary, open(os.path.join(a.out, "data_summary.json"), "w"), indent=1)
    tab = collections.defaultdict(collections.Counter)
    for (l, s, lab), n in cnt.items():
        tab[(l, s)]["multi" if "|" in lab else lab] += n
    for k in sorted(tab):
        print(k, dict(tab[k]))
    print("train rows dropped as dev/test leaks:", dict(dropped))


if __name__ == "__main__":
    main()

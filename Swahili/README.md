# Swahili Multidomain Data Pipeline

Collection, cleaning, review and domain classification of Swahili text for five domains: agriculture, finance, health, law and telecommunications. This is the Swahili track of the CMU-Africa MSIT practicum with Intron (offline multilingual spoken question answering). Author: Michael Sangwa.

Counts are **words** (whitespace-separated). No subword tokenizer has been chosen. Nothing here is labelled human-verified unless it appears in a section called "Human review".

## Status (7 October 2026)

| Domain | Licence-clear words | Share of 200k target |
|---|---:|---:|
| Agriculture | 206,813 | 103% |
| Finance | 301,005 | 151% |
| Health | 172,559 | 86% |
| Law | 112,207 | 56% |
| Telecoms | 190,288 | 95% |
| **Total** | **982,872** | **98% of 1M** (875,054 counting each domain at most 200k) |

Separate tier, not counted above: **381,292 words** of Swahili news (finance 196,130, health 185,162). The dataset cards say CC BY 4.0, but the rights of the underlying newspapers are not cleared, and glued sentences in the source text were repaired mechanically.

"Licence-clear" means the text passes the quality rules and comes from a source with a stated licence: CC BY documents, Wikipedia (CC BY-SA), public-domain VOA text, and FineWeb-2 (ODC-By; the client approved its use, written confirmation is pending). "Passes the rules" is not "human-verified": see Human review for the measured keep rates.

## Notebooks and scripts

| File | Purpose |
|---|---|
| `Swahili_Corpus_Audit_Colab.ipynb` | Pre-run notebook: cleaning funnel, caption problem, human review, licences, seed corpus, PnC annotation, classifier, word target by tier. Runs without uploads. |
| `swahili_final.zip` | Scripts, documentation and result tables (see below). |
| `classifier_report.md` | Domain classifier trained on the seed and tested on human-reviewed passages. |

Inside the zip, scripts run in this order: `build_deliverables.py` (shared JSONL, statistics, licence tiers), `fetch_wikipedia.py` and `process_wikipedia.py`, `extract_tari.py`, `extract_tanzlii_pdfs.py`, `fetch_voa_agri.py`, `process_fineweb_domains.py` and `process_fineweb_train.py`, `process_swahili_news.py` and `process_finance_news.py`, `annotate_pnc.py`, `build_seed.py`, `train_eval_seed.py`, `build_notebook.py`. Shared filters: `caption_filter.py`, `quality_rules.py`.

## Pipeline

1. **Sources** with a record of licence and evidence URL (`results/source_register.csv`).
2. **Caption filter.** Most AfriVoice Swahili text is people describing a picture. A rule-based detector (fitted to 100 human labels) removes it.
3. **Quality rules.** Reject passages with no terminal punctuation, web boilerplate, spaces before punctuation, or sentences glued together.
4. **Domain.** Passages need at least two domain keywords (per-domain lexicon); the larger count wins.
5. **Near-duplicates.** Word 8-gram Jaccard of 0.5 or more.
6. **Annotation.** Word-level punctuation (none, comma, period, question, exclamation, other) and capitalisation (lower, title, upper, mixed) labels, with a round-trip check that rebuilds each passage from its tokens.
7. **Seed corpus.** About 10,000 words per domain from licence-clear text (`results/seed_summary.json`).

Shared record format: `id, language, domain, raw_text, source_dataset, source_split, license, review_status, notes`.

## Human review (Michael)

| Review | n | Result |
|---|---:|---|
| First review: mixed candidate pool | 100 | 19 kept. Every one of the 54 AfriVoice passages failed (picture descriptions). |
| AfriVoice only, unfiltered | 100 | 2 kept; 74 domain right, 66 punctuation usable. Of these 100, 70 are not independently confirmed; see `results/afrivoice_review_scored.json`. |
| AfriVoice blind re-review | 30 | 0 kept; agrees with the earlier answers on 22 of 30 rows. |
| FineWeb-2 (agriculture, finance, telecoms) | 90 | **50 kept (55%)**; 64% domain right, 86% punctuation usable. Agriculture 17 of 30, finance 18 of 30, telecoms 15 of 30. |

About 9,000 human-verified words in total. This is below the 10,000 verified words per domain milestone, so none of the five domains meets it yet. AfriVoice is excluded for Swahili.

## Domain classifier

TF-IDF (word 1-2 grams) and class-balanced logistic regression trained on the seed, tested on the 97 human-reviewed passages whose domain was confirmed: **66.0%** (gold 84.2%). Law recall is 0.25. The score depends on the seed's composition: seeds built from parliament, forum and web text look different from the statutes and news in the test set. The earlier 95% figure measured agreement with keyword labels, not with human judgement. See `classifier_report.md`.

## Not in this repository

Corpus text from FineWeb-2, the news datasets, AfriVoice and TanzLII is not included. The repository is public and redistribution of these sources is not cleared; their identifiers, licences and counts are recorded instead. Included data: public-domain VOA text, CC BY TARI passages and CC BY-SA Wikipedia passages (attribute Wikipedia authors; the source article is in each record's notes).

## Limitations

- FineWeb-2 text is mostly web, news and parliament text; about 45% of it failed human review, mainly on domain.
- Questions are rare: 885 question marks in about 1,044,000 annotated words (0.08%), which matters for spoken question answering.
- Law has the largest gap (88k below 200k). TanzLII text needs a licence check and bulk access.
- One reviewer. A second reviewer and a fluent-speaker check are still needed.
- Telecoms and agriculture domain precision rests on keywords plus a 30-passage review per domain.

## Licences

CC BY 4.0 (TARI, dataset cards), CC BY-SA (Wikipedia), public domain (VOA-produced text; wire stories excluded), ODC-By (FineWeb-2). Each record carries its licence string.

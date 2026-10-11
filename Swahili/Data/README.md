# Swahili data statistics (words, whitespace-separated)

Files: `Data.csv` (licence-clear tier, columns `Language, Domain, Source, Text, Split, License`), `Evaluation.csv` (dev rows with `Text_No_Punct`, `Hypothesis`, `Domain_Pred`), `Data_news_tier.csv` (news, rights unresolved, kept apart). Language code: `swa`. Corpus text is not in the public repository; see the Swahili README for why.

## Words per domain and split

| Domain | Train words | Dev words | Train passages | Dev passages | Human-verified words (all reviews) |
|---|---:|---:|---:|---:|---:|
| agriculture | 204,550 | 1,178 | 3,036 | 18 | 1,276 |
| finance | 299,085 | 1,188 | 4,570 | 18 | 1,188 |
| health | 169,863 | 1,742 | 2,069 | 18 | 1,773 |
| law | 107,173 | 2,145 | 955 | 16 | 2,634 |
| telecoms | 184,470 | 4,397 | 2,236 | 24 | 4,758 |
| **Total** | **965,141** | **10,650** | 12,866 | 94 | **11,629** |

Dev = passages a human reviewer kept (94 passages). The dev split is small; more reviewed passages will be added. Human-verified words include reviewed passages that are not in the annotated set.

## Words per domain by source family (train and dev)

| Source family | agriculture | finance | health | law | telecoms |
|---|---:|---:|---:|---:|---:|
| Documents with a CC licence (TARI, IITA, guides, others) | 12,632 | 0 | 157 | 4,316 | 72 |
| FineWeb-2 (ODC-By, client approved) | 182,273 | 239,834 | 6,085 | 43,537 | 167,373 |
| TanzLII (CC BY, to confirm) | 0 | 0 | 0 | 27,680 | 0 |
| VOA Swahili (public domain) | 2,990 | 0 | 0 | 0 | 0 |
| Wikipedia (CC BY-SA) | 7,833 | 60,439 | 165,363 | 33,785 | 21,422 |

## News tier (separate file, not counted above)

| Domain | Words |
|---|---:|
| finance | 196,153 |
| health | 185,185 |

Dataset cards say CC BY 4.0, but newspaper rights are not cleared; glued sentences in the source were repaired mechanically.

## How the split was made

1. Dev: passages kept in a human review. 2. Train: every other licence-clear passage, except those a reviewer marked wrong domain or unusable punctuation and near-duplicates (word 8-gram Jaccard 0.5 or more) of a dev passage. 3. Exact duplicate texts were dropped. Script: `build_shared_format.py`.

## Caveats

- Passing the rules is not human verification; about 55% of FineWeb-2 passages, 60% of health and 43% of law passages (Wikipedia and TanzLII) were kept in review.
- FineWeb-2 use was approved verbally by the client; written confirmation is pending.
- TanzLII passages carry a CC BY licence recorded from the earlier source register; the PDFs print none and the site's licence note is still to be confirmed. In review, 4 of 13 TanzLII passages were kept; statute fragments often start or stop mid-section.

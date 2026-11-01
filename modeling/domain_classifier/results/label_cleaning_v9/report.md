# Label cleaning (confident learning on out-of-fold TF-IDF probabilities)

Data: `data_v9/domain_train.jsonl`, 187,621 rows. Flagged: 5,105; dropped (weak or teacher labels only): 4,648.

## Dropped rows by language and label

| Language | Given label | Rows (weak/teacher) | Dropped | Share | Most often suggested |
|---|---|---:|---:|---:|---|
| kin | agriculture | 14,505 | 385 | 2.7% | finance 213, other 118, health 54 |
| kin | finance | 14,521 | 937 | 6.5% | other 699, agriculture 172, health 59 |
| kin | health | 31,404 | 1,028 | 3.3% | other 709, agriculture 159, finance 155 |
| kin | law | 17,056 | 175 | 1.0% | other 99, health 32, finance 30 |
| kin | telecoms | 351 | 20 | 5.7% | other 14, finance 6 |
| kin | other | 48,127 | 1,736 | 3.6% | finance 1038, health 418, agriculture 245 |
| pcm | agriculture | 429 | 8 | 1.9% | other 8 |
| pcm | finance | 1,113 | 9 | 0.8% | other 8, agriculture 1 |
| pcm | health | 2,293 | 2 | 0.1% | other 1, finance 1 |
| pcm | law | 2,711 | 20 | 0.7% | other 14, finance 4, agriculture 1 |
| pcm | telecoms | 78 | 1 | 1.3% | other 1 |
| pcm | other | 11,020 | 40 | 0.4% | finance 19, health 11, law 6 |
| swa | agriculture | 4,027 | 57 | 1.4% | finance 28, health 20, other 6 |
| swa | finance | 9,912 | 107 | 1.1% | telecoms 39, agriculture 36, health 21 |
| swa | health | 4,865 | 34 | 0.7% | law 16, finance 9, telecoms 4 |
| swa | law | 1,301 | 35 | 2.7% | finance 21, telecoms 7, agriculture 3 |
| swa | telecoms | 2,455 | 23 | 0.9% | finance 10, other 5, agriculture 3 |
| swa | other | 4,415 | 9 | 0.2% | health 8, finance 1 |
| yor | agriculture | 65 | 0 | 0.0% |  |
| yor | finance | 10 | 1 | 10.0% | other 1 |
| yor | health | 319 | 0 | 0.0% |  |
| yor | law | 919 | 0 | 0.0% |  |
| yor | telecoms | 16 | 0 | 0.0% |  |
| yor | other | 2,890 | 21 | 0.7% | health 18, finance 2, law 1 |

## Dropped rows by source (sources with at least 100 weak/teacher rows)

| Language | Source | Rows | Dropped | Share |
|---|---|---:|---:|---:|
| kin | kinnews:technology | 112 | 18 | 16.1% |
| kin | kinnews:health | 1,334 | 205 | 15.4% |
| kin | kinnews:economy | 451 | 62 | 13.7% |
| kin | kinnews:politic | 1,728 | 141 | 8.2% |
| kin | afrivoice_kin:government:val | 426 | 32 | 7.5% |
| kin | afrivoice_kin:government:test | 349 | 25 | 7.2% |
| kin | afrivoice_kin:financial:val | 1,007 | 65 | 6.5% |
| kin | kinnews:tourism | 110 | 6 | 5.5% |
| kin | afrivoice_kin:health:val | 1,384 | 75 | 5.4% |
| kin | DigitalUmuganda/Afrivoice_Kinyarwanda | 32,726 | 1,681 | 5.1% |
| kin | afrivoice_kin:government:train | 23,299 | 1,177 | 5.1% |
| kin | afrivoice_kin:health:test | 1,569 | 77 | 4.9% |
| kin | afrivoice_kin:financial:test | 2,283 | 95 | 4.2% |
| kin | kinnews:relationship | 682 | 25 | 3.7% |
| swa | health-speech-2026-27.pdf | 114 | 3 | 2.6% |
| kin | afrivoice_kin:agriculture:test | 2,109 | 52 | 2.5% |
| kin | kinnews:history | 213 | 5 | 2.3% |
| kin | afrivoice_kin:scripted_education:train | 7,408 | 170 | 2.3% |
| swa | HuggingFaceFW/fineweb-2 | 527 | 11 | 2.1% |
| kin | afrivoice_kin:agriculture:val | 1,038 | 19 | 1.8% |
| swa | Mollel/SwahiliNewsClassification (uchumi) | 1,829 | 33 | 1.8% |
| kin | afrivoice_kin:education:train | 9,003 | 161 | 1.8% |
| kin | kinnews:education | 183 | 3 | 1.6% |
| kin | kinnews:fashion | 126 | 2 | 1.6% |
| kin | kinnews:entertainment | 1,831 | 29 | 1.6% |
| pcm | mafand_en_pcm | 369 | 5 | 1.4% |
| yor | fineweb2:www.bbc.com | 679 | 9 | 1.3% |
| kin | afrivoice_kin:education:test | 396 | 5 | 1.3% |
| swa | naot-central-2022-23.pdf | 169 | 2 | 1.2% |
| kin | kinnews:religion | 685 | 8 | 1.2% |
| swa | HuggingFaceFW/fineweb-2 swh_Latn | 8,055 | 85 | 1.1% |
| yor | fineweb2:iroyinowuro.com.ng | 401 | 4 | 1.0% |
| swa | TanzLII PDF: tz-act-2023-10-publication-document (1).pdf | 205 | 2 | 1.0% |
| kin | kinnews:culture | 207 | 2 | 1.0% |
| kin | KINNEWS | 239 | 2 | 0.8% |
| swa | DigitalUmuganda/Afrivoice_Swahili | 4,512 | 36 | 0.8% |
| kin | afrivoice_kin:education:val | 131 | 1 | 0.8% |
| yor | fineweb2:www.jw.org | 427 | 3 | 0.7% |
| kin | DigitalUmuganda/NMT_Rwandan-Gazette_parallel_data_en_kin | 16,728 | 110 | 0.7% |
| pcm | fineweb2:pidginbible.com | 498 | 3 | 0.6% |
| kin | kinnews:sport | 1,678 | 9 | 0.5% |
| pcm | wikipedia_pcm | 407 | 2 | 0.5% |
| pcm | xlsum_pidgin | 13,994 | 68 | 0.5% |
| yor | fineweb2:www.biblica.com | 223 | 1 | 0.4% |
| swa | mollel_news:kitaifa | 1,365 | 6 | 0.4% |
| swa | community-datasets/swahili_news | 1,827 | 7 | 0.4% |
| swa | mollel_news:burudani | 798 | 2 | 0.3% |
| pcm | fineweb2:www.bbc.com | 421 | 1 | 0.2% |
| kin | digitalumuganda:nmt_health:rbc | 7,537 | 15 | 0.2% |
| yor | wura:alaroye.org | 508 | 1 | 0.2% |
| yor | fineweb2:alaroye.org | 660 | 1 | 0.2% |
| swa | mollel_news:kimataifa | 793 | 1 | 0.1% |
| kin | digitalumuganda:rbc_health | 8,976 | 4 | 0.0% |
| swa | TARI/IITA rice training guide (tari.go.tz) | 115 | 0 | 0.0% |
| swa | mollel_news:michezo | 1,459 | 0 | 0.0% |
| pcm | fineweb2_pcm | 288 | 0 | 0.0% |
| yor | wura:www.bbc.com | 276 | 0 | 0.0% |
| pcm | fineweb2:pcm.wikipedia.org | 216 | 0 | 0.0% |
| yor | fineweb2:radionigeriaibadan.gov.ng | 102 | 0 | 0.0% |
| yor | fineweb2:iweirohinapere.com | 108 | 0 | 0.0% |
| yor | fineweb2:alaroye.tv | 144 | 0 | 0.0% |

## Self-confidence thresholds

```
{
 "kin": {
  "agriculture": 0.9120000004768372,
  "finance": 0.8009999990463257,
  "health": 0.8759999871253967,
  "law": 0.9570000171661377,
  "telecoms": 0.7990000247955322,
  "other": 0.8339999914169312
 },
 "pcm": {
  "agriculture": 0.7870000004768372,
  "finance": 0.8040000200271606,
  "health": 0.8930000066757202,
  "law": 0.8989999890327454,
  "telecoms": 0.5630000233650208,
  "other": 0.9440000057220459
 },
 "swa": {
  "agriculture": 0.9110000133514404,
  "finance": 0.9340000152587891,
  "health": 0.9089999794960022,
  "law": 0.871999979019165,
  "telecoms": 0.9580000042915344,
  "other": 0.8550000190734863
 },
 "yor": {
  "agriculture": 0.699999988079071,
  "finance": 0.13699999451637268,
  "health": 0.7509999871253967,
  "law": 0.9599999785423279,
  "telecoms": 0.460999995470047,
  "other": 0.953000009059906
 }
}
```

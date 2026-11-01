# Label cleaning (confident learning on out-of-fold TF-IDF probabilities)

Data: `data_v8/domain_train.jsonl`, 181,975 rows. Flagged: 5,070; dropped (weak or teacher labels only): 4,629.

## Dropped rows by language and label

| Language | Given label | Rows (weak/teacher) | Dropped | Share | Most often suggested |
|---|---|---:|---:|---:|---|
| kin | agriculture | 14,505 | 386 | 2.7% | finance 210, other 123, health 53 |
| kin | finance | 14,521 | 944 | 6.5% | other 708, agriculture 171, health 58 |
| kin | health | 31,404 | 1,048 | 3.3% | other 730, agriculture 157, finance 156 |
| kin | law | 17,056 | 181 | 1.1% | other 101, health 35, finance 31 |
| kin | telecoms | 351 | 25 | 7.1% | other 17, finance 8 |
| kin | other | 48,127 | 1,704 | 3.5% | finance 1015, health 413, agriculture 244 |
| pcm | agriculture | 429 | 11 | 2.6% | other 10, finance 1 |
| pcm | finance | 1,111 | 11 | 1.0% | other 7, agriculture 2, telecoms 1 |
| pcm | health | 2,289 | 1 | 0.0% | finance 1 |
| pcm | law | 2,708 | 18 | 0.7% | other 13, finance 4, health 1 |
| pcm | telecoms | 78 | 1 | 1.3% | other 1 |
| pcm | other | 8,524 | 35 | 0.4% | finance 16, health 12, law 3 |
| swa | agriculture | 4,027 | 46 | 1.1% | finance 22, health 17, other 4 |
| swa | finance | 9,912 | 111 | 1.1% | agriculture 38, telecoms 38, health 25 |
| swa | health | 4,865 | 36 | 0.7% | law 17, finance 9, telecoms 5 |
| swa | law | 1,301 | 32 | 2.5% | finance 20, telecoms 6, other 2 |
| swa | telecoms | 2,455 | 23 | 0.9% | finance 10, other 7, agriculture 3 |
| swa | other | 4,415 | 13 | 0.3% | health 8, finance 5 |
| yor | agriculture | 47 | 0 | 0.0% |  |
| yor | finance | 8 | 0 | 0.0% |  |
| yor | health | 258 | 0 | 0.0% |  |
| yor | law | 754 | 3 | 0.4% | other 3 |
| yor | telecoms | 11 | 0 | 0.0% |  |

## Dropped rows by source (sources with at least 100 weak/teacher rows)

| Language | Source | Rows | Dropped | Share |
|---|---|---:|---:|---:|
| kin | kinnews:technology | 112 | 21 | 18.8% |
| kin | kinnews:health | 1,334 | 209 | 15.7% |
| kin | kinnews:economy | 451 | 61 | 13.5% |
| kin | kinnews:politic | 1,728 | 144 | 8.3% |
| kin | afrivoice_kin:government:val | 426 | 32 | 7.5% |
| kin | afrivoice_kin:government:test | 349 | 25 | 7.2% |
| kin | afrivoice_kin:financial:val | 1,007 | 69 | 6.9% |
| kin | kinnews:tourism | 110 | 6 | 5.5% |
| kin | afrivoice_kin:health:val | 1,384 | 75 | 5.4% |
| kin | afrivoice_kin:health:test | 1,569 | 82 | 5.2% |
| kin | DigitalUmuganda/Afrivoice_Kinyarwanda | 32,726 | 1,697 | 5.2% |
| kin | afrivoice_kin:government:train | 23,299 | 1,149 | 4.9% |
| kin | afrivoice_kin:financial:test | 2,283 | 96 | 4.2% |
| kin | kinnews:relationship | 682 | 25 | 3.7% |
| swa | health-speech-2026-27.pdf | 114 | 3 | 2.6% |
| kin | kinnews:fashion | 126 | 3 | 2.4% |
| kin | afrivoice_kin:agriculture:test | 2,109 | 50 | 2.4% |
| kin | kinnews:history | 213 | 5 | 2.3% |
| kin | afrivoice_kin:scripted_education:train | 7,408 | 161 | 2.2% |
| kin | afrivoice_kin:agriculture:val | 1,038 | 20 | 1.9% |
| swa | HuggingFaceFW/fineweb-2 | 527 | 10 | 1.9% |
| kin | afrivoice_kin:education:train | 9,003 | 168 | 1.9% |
| swa | Mollel/SwahiliNewsClassification (uchumi) | 1,829 | 34 | 1.9% |
| kin | KINNEWS | 239 | 4 | 1.7% |
| kin | kinnews:education | 183 | 3 | 1.6% |
| kin | kinnews:entertainment | 1,831 | 27 | 1.5% |
| kin | afrivoice_kin:education:test | 396 | 5 | 1.3% |
| swa | naot-central-2022-23.pdf | 169 | 2 | 1.2% |
| swa | HuggingFaceFW/fineweb-2 swh_Latn | 8,055 | 80 | 1.0% |
| swa | TanzLII PDF: tz-act-2023-10-publication-document (1).pdf | 205 | 2 | 1.0% |
| kin | kinnews:culture | 207 | 2 | 1.0% |
| kin | kinnews:religion | 685 | 6 | 0.9% |
| pcm | mafand_en_pcm | 369 | 3 | 0.8% |
| kin | afrivoice_kin:education:val | 131 | 1 | 0.8% |
| swa | DigitalUmuganda/Afrivoice_Swahili | 4,512 | 34 | 0.8% |
| yor | wura:www.bbc.com | 276 | 2 | 0.7% |
| kin | DigitalUmuganda/NMT_Rwandan-Gazette_parallel_data_en_kin | 16,728 | 113 | 0.7% |
| kin | kinnews:sport | 1,678 | 10 | 0.6% |
| swa | mollel_news:kitaifa | 1,365 | 8 | 0.6% |
| pcm | xlsum_pidgin | 13,994 | 74 | 0.5% |
| swa | community-datasets/swahili_news | 1,827 | 6 | 0.3% |
| swa | mollel_news:burudani | 798 | 2 | 0.3% |
| kin | digitalumuganda:nmt_health:rbc | 7,537 | 14 | 0.2% |
| swa | mollel_news:michezo | 1,459 | 2 | 0.1% |
| swa | mollel_news:kimataifa | 793 | 1 | 0.1% |
| kin | digitalumuganda:rbc_health | 8,976 | 5 | 0.1% |
| swa | TARI/IITA rice training guide (tari.go.tz) | 115 | 0 | 0.0% |
| pcm | fineweb2_pcm | 288 | 0 | 0.0% |
| pcm | wikipedia_pcm | 407 | 0 | 0.0% |
| yor | wura:alaroye.org | 508 | 0 | 0.0% |

## Self-confidence thresholds

```
{
 "kin": {
  "agriculture": 0.9110000133514404,
  "finance": 0.7979999780654907,
  "health": 0.8740000128746033,
  "law": 0.9570000171661377,
  "telecoms": 0.7979999780654907,
  "other": 0.8389999866485596
 },
 "pcm": {
  "agriculture": 0.7929999828338623,
  "finance": 0.8090000152587891,
  "health": 0.8970000147819519,
  "law": 0.9020000100135803,
  "telecoms": 0.5669999718666077,
  "other": 0.9399999976158142
 },
 "swa": {
  "agriculture": 0.9089999794960022,
  "finance": 0.9330000281333923,
  "health": 0.9070000052452087,
  "law": 0.8679999709129333,
  "telecoms": 0.9570000171661377,
  "other": 0.8569999933242798
 },
 "yor": {
  "agriculture": 0.6200000047683716,
  "finance": 0.12800000607967377,
  "health": 0.7839999794960022,
  "law": 0.9570000171661377,
  "telecoms": 0.22300000488758087,
  "other": 0.9340000152587891
 }
}
```

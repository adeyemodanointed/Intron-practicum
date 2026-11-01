# Multilingual domain classifier: results

One model for every language. Labels: agriculture, finance, health, law, telecoms; a text gets one or two domains, or `other` when no domain reaches its threshold.

Model: word 1-2 gram + character 3-5 gram TF-IDF (lower-cased, diacritics removed, language tag token) and one class-balanced logistic regression per domain. Test sets are human-labelled except Yoruba (synthetic, unreviewed); `ALL` pools the human-labelled rows only.

## Data

| Language | Train rows | Dev rows (threshold tuning) | Test rows |
|---|---:|---:|---:|
| kin | 49,809 | 189 | 385 |
| pcm | 19,409 | 667 | 1,533 |
| swa | 20,863 | 214 | 240 |
| yor | 5,966 | 1,500 | 1,500 |

Thresholds (shared, then per-language overrides tuned on that language's dev set):

- all: agriculture 0.375, finance 0.775, health 0.7, law 0.575, telecoms 0.65
- kin: agriculture 0.75, finance 0.825, health 0.7, law 0.725, telecoms 0.05
- pcm: finance 0.625, health 0.3
- swa: agriculture 0.7, finance 0.9, health 0.4, law 0.1, telecoms 0.375
- yor: agriculture 0.55, finance 0.55, health 0.575, law 0.425, telecoms 0.775

## Summary

| Setup | Language | Test n | Primary label correct | Exact label set | Macro F1 | Labels with test data | Human multi-domain | Predicted multi-domain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| one model | kin | 385 | 73.2% | 72.5% | 0.672 | 6 of 6 | 0 | 7 |
| one model | pcm | 1533 | 86.0% | 86.0% | 0.751 | 3 of 6 | 0 | 1 |
| one model | swa | 240 | 73.3% | 70.8% | 0.751 | 6 of 6 | 9 | 2 |
| one model | yor | 1500 | 95.2% | 93.4% | 0.958 | 5 of 6 | 0 | 45 |
| one model | ALL | 2158 | 82.3% | 81.9% | 0.767 | 6 of 6 | 9 | 10 |
| per-language model | kin | 385 | 70.1% | 69.6% | 0.636 | 6 of 6 | 0 | 3 |
| per-language model | pcm | 1533 | 85.6% | 85.6% | 0.754 | 3 of 6 | 0 | 2 |
| per-language model | swa | 240 | 75.4% | 72.5% | 0.765 | 6 of 6 | 9 | 4 |
| per-language model | yor | 1500 | 95.0% | 92.8% | 0.957 | 5 of 6 | 0 | 52 |

## One model, kin

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.800 | 0.882 | 0.839 | 68 | 75 |
| finance | 0.641 | 0.745 | 0.689 | 55 | 64 |
| health | 0.583 | 0.754 | 0.658 | 65 | 84 |
| law | 0.871 | 0.983 | 0.924 | 117 | 132 |
| telecoms | 0.727 | 0.615 | 0.667 | 13 | 11 |
| other | 0.462 | 0.179 | 0.258 | 67 | 26 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 59 | 1 | 5 | 0 | 0 | 3 |
| finance | 3 | 41 | 1 | 5 | 0 | 5 |
| health | 7 | 3 | 48 | 2 | 0 | 5 |
| law | 0 | 0 | 1 | 115 | 0 | 1 |
| telecoms | 0 | 5 | 0 | 1 | 7 | 0 |
| other | 5 | 14 | 26 | 9 | 1 | 12 |

## One model, pcm

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.000 | 0.000 | 0.000 | 0 | 0 |
| finance | 0.880 | 0.432 | 0.579 | 169 | 83 |
| health | 0.754 | 0.759 | 0.757 | 170 | 171 |
| law | 0.000 | 0.000 | 0.000 | 0 | 35 |
| telecoms | 0.000 | 0.000 | 0.000 | 0 | 0 |
| other | 0.897 | 0.936 | 0.916 | 1194 | 1245 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 0 | 0 | 0 | 0 | 0 | 0 |
| finance | 0 | 73 | 9 | 0 | 0 | 87 |
| health | 0 | 0 | 129 | 0 | 0 | 41 |
| law | 0 | 0 | 0 | 0 | 0 | 0 |
| telecoms | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 10 | 32 | 35 | 0 | 1117 |

## One model, swa

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.875 | 0.824 | 0.848 | 34 | 32 |
| finance | 0.839 | 0.722 | 0.776 | 36 | 31 |
| health | 0.938 | 0.789 | 0.857 | 38 | 32 |
| law | 0.738 | 0.816 | 0.775 | 38 | 42 |
| telecoms | 0.609 | 0.848 | 0.709 | 33 | 46 |
| other | 0.593 | 0.500 | 0.543 | 70 | 59 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 28 | 0 | 0 | 0 | 0 | 6 |
| finance | 0 | 26 | 0 | 0 | 1 | 6 |
| health | 0 | 1 | 30 | 0 | 0 | 7 |
| law | 0 | 0 | 0 | 29 | 2 | 4 |
| telecoms | 0 | 1 | 0 | 0 | 28 | 1 |
| other | 4 | 3 | 2 | 11 | 15 | 35 |

## One model, yor

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.979 | 0.953 | 0.966 | 300 | 292 |
| finance | 0.899 | 0.950 | 0.924 | 300 | 317 |
| health | 0.970 | 0.983 | 0.977 | 300 | 304 |
| law | 0.954 | 0.967 | 0.960 | 300 | 304 |
| telecoms | 0.960 | 0.967 | 0.963 | 300 | 302 |
| other | 0.000 | 0.000 | 0.000 | 0 | 26 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 284 | 5 | 6 | 0 | 0 | 5 |
| finance | 3 | 277 | 0 | 6 | 9 | 5 |
| health | 1 | 0 | 294 | 1 | 0 | 4 |
| law | 0 | 7 | 1 | 285 | 2 | 5 |
| telecoms | 0 | 4 | 0 | 1 | 288 | 7 |
| other | 0 | 0 | 0 | 0 | 0 | 0 |

## One model, ALL

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.822 | 0.863 | 0.842 | 102 | 107 |
| finance | 0.787 | 0.538 | 0.639 | 260 | 178 |
| health | 0.725 | 0.762 | 0.743 | 273 | 287 |
| law | 0.699 | 0.942 | 0.802 | 155 | 209 |
| telecoms | 0.632 | 0.783 | 0.699 | 46 | 57 |
| other | 0.875 | 0.875 | 0.875 | 1331 | 1330 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 87 | 1 | 5 | 0 | 0 | 9 |
| finance | 3 | 140 | 10 | 5 | 1 | 98 |
| health | 7 | 4 | 207 | 2 | 0 | 53 |
| law | 0 | 0 | 1 | 144 | 2 | 5 |
| telecoms | 0 | 6 | 0 | 1 | 35 | 1 |
| other | 9 | 27 | 60 | 55 | 16 | 1164 |

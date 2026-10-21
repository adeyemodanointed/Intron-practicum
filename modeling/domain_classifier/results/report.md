# Multilingual domain classifier: results

One model for every language. Labels: agriculture, finance, health, law, telecoms; a text gets one or two domains, or `other` when no domain reaches its threshold.

Model: word 1-2 gram + character 3-5 gram TF-IDF (lower-cased, diacritics removed, language tag token) and one class-balanced logistic regression per domain. Scores below are on human-labelled test sets only.

## Data

| Language | Train rows | Dev rows (threshold tuning) | Test rows (human labels) |
|---|---:|---:|---:|
| kin | 49,809 | 189 | 385 |
| pcm | 19,409 | 667 | 1,533 |
| swa | 20,863 | 214 | 240 |

Thresholds (shared, then per-language overrides tuned on that language's dev set):

- all: agriculture 0.7, finance 0.825, health 0.6, law 0.825, telecoms 0.125
- kin: agriculture 0.75, finance 0.925, health 0.7, law 0.725, telecoms 0.05
- pcm: finance 0.4, health 0.275
- swa: agriculture 0.7, finance 0.825, health 0.375, law 0.225, telecoms 0.125

## Summary

| Setup | Language | Test n | Primary label correct | Exact label set | Macro F1 | Labels with test data | Human multi-domain | Predicted multi-domain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| one model | kin | 385 | 71.2% | 70.6% | 0.659 | 6 of 6 | 0 | 7 |
| one model | pcm | 1533 | 87.5% | 87.3% | 0.782 | 3 of 6 | 0 | 3 |
| one model | swa | 240 | 70.8% | 67.5% | 0.726 | 6 of 6 | 9 | 3 |
| one model | ALL | 2158 | 82.7% | 82.1% | 0.764 | 6 of 6 | 9 | 13 |
| per-language model | kin | 385 | 70.1% | 69.6% | 0.636 | 6 of 6 | 0 | 3 |
| per-language model | pcm | 1533 | 85.6% | 85.6% | 0.754 | 3 of 6 | 0 | 2 |
| per-language model | swa | 240 | 75.4% | 72.5% | 0.765 | 6 of 6 | 9 | 4 |

## One model, kin

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.800 | 0.882 | 0.839 | 68 | 75 |
| finance | 0.646 | 0.564 | 0.602 | 55 | 48 |
| health | 0.578 | 0.738 | 0.649 | 65 | 83 |
| law | 0.871 | 0.983 | 0.924 | 117 | 132 |
| telecoms | 0.727 | 0.615 | 0.667 | 13 | 11 |
| other | 0.349 | 0.224 | 0.273 | 67 | 43 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 59 | 0 | 5 | 0 | 0 | 4 |
| finance | 3 | 31 | 1 | 5 | 0 | 15 |
| health | 7 | 2 | 47 | 2 | 0 | 7 |
| law | 0 | 0 | 1 | 115 | 0 | 1 |
| telecoms | 0 | 4 | 0 | 1 | 7 | 1 |
| other | 5 | 11 | 26 | 9 | 1 | 15 |

## One model, pcm

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.000 | 0.000 | 0.000 | 0 | 0 |
| finance | 0.814 | 0.568 | 0.669 | 169 | 118 |
| health | 0.750 | 0.759 | 0.754 | 170 | 172 |
| law | 0.000 | 0.000 | 0.000 | 0 | 21 |
| telecoms | 0.000 | 0.000 | 0.000 | 0 | 4 |
| other | 0.914 | 0.935 | 0.924 | 1194 | 1221 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 0 | 0 | 0 | 0 | 0 | 0 |
| finance | 0 | 96 | 6 | 0 | 3 | 64 |
| health | 0 | 0 | 129 | 0 | 0 | 41 |
| law | 0 | 0 | 0 | 0 | 0 | 0 |
| telecoms | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 22 | 34 | 21 | 1 | 1116 |

## One model, swa

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.875 | 0.824 | 0.848 | 34 | 32 |
| finance | 0.824 | 0.778 | 0.800 | 36 | 34 |
| health | 0.939 | 0.816 | 0.873 | 38 | 33 |
| law | 0.781 | 0.658 | 0.714 | 38 | 32 |
| telecoms | 0.500 | 0.879 | 0.637 | 33 | 58 |
| other | 0.556 | 0.429 | 0.484 | 70 | 54 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 28 | 0 | 0 | 0 | 0 | 6 |
| finance | 0 | 28 | 0 | 0 | 1 | 4 |
| health | 0 | 1 | 31 | 0 | 0 | 6 |
| law | 0 | 0 | 0 | 25 | 3 | 7 |
| telecoms | 0 | 1 | 0 | 0 | 28 | 1 |
| other | 4 | 4 | 2 | 7 | 23 | 30 |

## One model, ALL

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.822 | 0.863 | 0.842 | 102 | 107 |
| finance | 0.775 | 0.596 | 0.674 | 260 | 200 |
| health | 0.722 | 0.762 | 0.742 | 273 | 288 |
| law | 0.757 | 0.903 | 0.824 | 155 | 185 |
| telecoms | 0.507 | 0.804 | 0.622 | 46 | 73 |
| other | 0.881 | 0.872 | 0.877 | 1331 | 1318 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 87 | 0 | 5 | 0 | 0 | 10 |
| finance | 3 | 155 | 7 | 5 | 4 | 83 |
| health | 7 | 3 | 207 | 2 | 0 | 54 |
| law | 0 | 0 | 1 | 140 | 3 | 8 |
| telecoms | 0 | 5 | 0 | 1 | 35 | 2 |
| other | 9 | 37 | 62 | 37 | 25 | 1161 |

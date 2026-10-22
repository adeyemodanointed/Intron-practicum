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

- all: agriculture 0.725, finance 0.85, health 0.7, law 0.55, telecoms 0.1
- kin: agriculture 0.725, finance 0.95, health 0.625, law 0.75, telecoms 0.05
- pcm: finance 0.65, health 0.25
- swa: agriculture 0.675, finance 0.85, health 0.25, law 0.1, telecoms 0.1

## Summary

| Setup | Language | Test n | Primary label correct | Exact label set | Macro F1 | Labels with test data | Human multi-domain | Predicted multi-domain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| one model | kin | 385 | 69.9% | 69.4% | 0.640 | 6 of 6 | 0 | 5 |
| one model | pcm | 1533 | 86.6% | 86.6% | 0.761 | 3 of 6 | 0 | 1 |
| one model | swa | 240 | 72.5% | 69.6% | 0.741 | 6 of 6 | 9 | 4 |
| one model | ALL | 2158 | 82.0% | 81.6% | 0.754 | 6 of 6 | 9 | 10 |
| per-language model | kin | 385 | 70.1% | 69.6% | 0.636 | 6 of 6 | 0 | 3 |
| per-language model | pcm | 1533 | 85.6% | 85.6% | 0.754 | 3 of 6 | 0 | 2 |
| per-language model | swa | 240 | 75.4% | 72.5% | 0.765 | 6 of 6 | 9 | 4 |

## One model, kin

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.800 | 0.882 | 0.839 | 68 | 75 |
| finance | 0.659 | 0.527 | 0.586 | 55 | 44 |
| health | 0.562 | 0.769 | 0.649 | 65 | 89 |
| law | 0.871 | 0.983 | 0.924 | 117 | 132 |
| telecoms | 0.778 | 0.538 | 0.636 | 13 | 9 |
| other | 0.268 | 0.164 | 0.204 | 67 | 41 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 59 | 0 | 5 | 0 | 0 | 4 |
| finance | 3 | 29 | 1 | 5 | 0 | 17 |
| health | 7 | 1 | 48 | 2 | 0 | 7 |
| law | 0 | 0 | 1 | 115 | 0 | 1 |
| telecoms | 0 | 4 | 0 | 1 | 7 | 1 |
| other | 5 | 10 | 31 | 9 | 1 | 11 |

## One model, pcm

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.000 | 0.000 | 0.000 | 0 | 0 |
| finance | 0.904 | 0.444 | 0.595 | 169 | 83 |
| health | 0.759 | 0.776 | 0.767 | 170 | 174 |
| law | 0.000 | 0.000 | 0.000 | 0 | 32 |
| telecoms | 0.000 | 0.000 | 0.000 | 0 | 4 |
| other | 0.902 | 0.938 | 0.920 | 1194 | 1241 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 0 | 0 | 0 | 0 | 0 | 0 |
| finance | 0 | 75 | 8 | 0 | 3 | 83 |
| health | 0 | 0 | 132 | 0 | 0 | 38 |
| law | 0 | 0 | 0 | 0 | 0 | 0 |
| telecoms | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 0 | 8 | 33 | 32 | 1 | 1120 |

## One model, swa

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.875 | 0.824 | 0.848 | 34 | 32 |
| finance | 0.824 | 0.778 | 0.800 | 36 | 34 |
| health | 0.919 | 0.895 | 0.907 | 38 | 37 |
| law | 0.778 | 0.737 | 0.757 | 38 | 36 |
| telecoms | 0.509 | 0.879 | 0.644 | 33 | 57 |
| other | 0.604 | 0.414 | 0.492 | 70 | 48 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 28 | 0 | 0 | 0 | 0 | 6 |
| finance | 0 | 28 | 0 | 0 | 1 | 4 |
| health | 0 | 1 | 34 | 0 | 0 | 3 |
| law | 0 | 0 | 0 | 27 | 3 | 5 |
| telecoms | 0 | 1 | 0 | 0 | 28 | 1 |
| other | 4 | 4 | 3 | 8 | 22 | 29 |

## One model, ALL

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.822 | 0.863 | 0.842 | 102 | 107 |
| finance | 0.820 | 0.508 | 0.627 | 260 | 161 |
| health | 0.720 | 0.791 | 0.754 | 273 | 300 |
| law | 0.715 | 0.923 | 0.806 | 155 | 200 |
| telecoms | 0.514 | 0.783 | 0.621 | 46 | 70 |
| other | 0.872 | 0.872 | 0.872 | 1331 | 1330 |

Top prediction vs human label (rows: human, columns: predicted):

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 87 | 0 | 5 | 0 | 0 | 10 |
| finance | 3 | 132 | 9 | 5 | 4 | 104 |
| health | 7 | 2 | 214 | 2 | 0 | 48 |
| law | 0 | 0 | 1 | 142 | 3 | 6 |
| telecoms | 0 | 5 | 0 | 1 | 35 | 2 |
| other | 9 | 22 | 67 | 49 | 24 | 1160 |

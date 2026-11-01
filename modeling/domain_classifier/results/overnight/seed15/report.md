# AfroXLMR domain classifier: test results

Model `Davlan/afro-xlmr-base`, best checkpoint at epoch 2.50 of 4 epochs (dev mean macro F1 0.811); max length 256, batch 16, lr 2e-05.
Training rows 94,205 (at most 15000 per language and label), dev 2,570, test 3,658 (human labels).

Thresholds (shared, then per-language overrides):

- all: agriculture 0.9, finance 0.7, health 0.95, law 0.95, telecoms 0.05
- kin: agriculture 0.9, finance 0.25, health 0.95, law 0.05, telecoms 0.05
- pcm: finance 0.075, health 0.2
- swa: agriculture 0.075, finance 0.85, health 0.95, law 0.2, telecoms 0.05
- yor: agriculture 0.95, finance 0.35, health 0.925, law 0.65, telecoms 0.525

| Language | Test n | Top label correct | Exact label set | Macro F1 | Labels with test data | Human multi-domain | Predicted multi-domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| kin | 385 | 71.9% | 71.4% | 0.622 | 6 of 6 | 0 | 2 |
| pcm | 1533 | 88.7% | 88.6% | 0.810 | 3 of 6 | 0 | 2 |
| swa | 240 | 75.4% | 72.1% | 0.764 | 6 of 6 | 9 | 3 |
| yor | 1500 | 94.9% | 94.7% | 0.952 | 5 of 6 | 0 | 5 |
| ALL | 2158 | 84.2% | 83.7% | 0.778 | 6 of 6 | 9 | 7 |

## kin

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.747 | 0.912 | 0.821 | 68 | 83 |
| finance | 0.564 | 0.800 | 0.662 | 55 | 78 |
| health | 0.608 | 0.738 | 0.667 | 65 | 79 |
| law | 0.858 | 0.983 | 0.916 | 117 | 134 |
| telecoms | 0.778 | 0.538 | 0.636 | 13 | 9 |
| other | 0.250 | 0.015 | 0.028 | 67 | 4 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 62 | 1 | 5 | 0 | 0 | 0 |
| finance | 3 | 44 | 1 | 6 | 0 | 1 |
| health | 7 | 7 | 48 | 1 | 0 | 2 |
| law | 0 | 0 | 2 | 115 | 0 | 0 |
| telecoms | 0 | 5 | 0 | 1 | 7 | 0 |
| other | 11 | 21 | 23 | 10 | 1 | 1 |

## pcm

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.000 | 0.000 | 0.000 | 0 | 8 |
| finance | 0.789 | 0.663 | 0.720 | 169 | 142 |
| health | 0.788 | 0.765 | 0.776 | 170 | 165 |
| law | 0.000 | 0.000 | 0.000 | 0 | 18 |
| telecoms | 0.000 | 0.000 | 0.000 | 0 | 4 |
| other | 0.933 | 0.936 | 0.935 | 1194 | 1198 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 0 | 0 | 0 | 0 | 0 | 0 |
| finance | 2 | 112 | 5 | 0 | 4 | 46 |
| health | 4 | 1 | 130 | 1 | 0 | 34 |
| law | 0 | 0 | 0 | 0 | 0 | 0 |
| telecoms | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 2 | 28 | 29 | 17 | 0 | 1118 |

## swa

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.825 | 0.971 | 0.892 | 34 | 40 |
| finance | 0.756 | 0.944 | 0.840 | 36 | 45 |
| health | 0.944 | 0.895 | 0.919 | 38 | 36 |
| law | 0.703 | 0.684 | 0.693 | 38 | 37 |
| telecoms | 0.605 | 0.788 | 0.684 | 33 | 43 |
| other | 0.738 | 0.443 | 0.554 | 70 | 42 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 33 | 0 | 0 | 0 | 0 | 0 |
| finance | 0 | 34 | 0 | 0 | 1 | 1 |
| health | 1 | 1 | 34 | 0 | 0 | 2 |
| law | 0 | 0 | 0 | 26 | 1 | 8 |
| telecoms | 0 | 5 | 0 | 0 | 23 | 0 |
| other | 6 | 5 | 2 | 11 | 15 | 31 |

## yor

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.972 | 0.933 | 0.952 | 300 | 288 |
| finance | 0.894 | 0.957 | 0.924 | 300 | 321 |
| health | 0.967 | 0.963 | 0.965 | 300 | 299 |
| law | 0.962 | 0.940 | 0.951 | 300 | 293 |
| telecoms | 0.980 | 0.957 | 0.968 | 300 | 293 |
| other | 0.000 | 0.000 | 0.000 | 0 | 11 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 280 | 10 | 5 | 1 | 0 | 4 |
| finance | 3 | 287 | 2 | 4 | 4 | 0 |
| health | 4 | 0 | 289 | 2 | 1 | 4 |
| law | 1 | 11 | 3 | 282 | 1 | 2 |
| telecoms | 0 | 10 | 0 | 3 | 286 | 1 |
| other | 0 | 0 | 0 | 0 | 0 | 0 |

## ALL

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.725 | 0.931 | 0.815 | 102 | 131 |
| finance | 0.717 | 0.731 | 0.724 | 260 | 265 |
| health | 0.757 | 0.777 | 0.767 | 273 | 280 |
| law | 0.746 | 0.910 | 0.820 | 155 | 189 |
| telecoms | 0.589 | 0.717 | 0.647 | 46 | 56 |
| other | 0.924 | 0.864 | 0.893 | 1331 | 1244 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 95 | 1 | 5 | 0 | 0 | 0 |
| finance | 5 | 190 | 6 | 6 | 5 | 48 |
| health | 12 | 9 | 212 | 2 | 0 | 38 |
| law | 0 | 0 | 2 | 141 | 1 | 8 |
| telecoms | 0 | 10 | 0 | 1 | 30 | 0 |
| other | 19 | 54 | 54 | 38 | 16 | 1150 |

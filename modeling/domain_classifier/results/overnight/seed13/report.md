# AfroXLMR domain classifier: test results

Model `Davlan/afro-xlmr-base`, best checkpoint at epoch 0.50 of 4 epochs (dev mean macro F1 0.808); max length 256, batch 16, lr 2e-05.
Training rows 94,205 (at most 15000 per language and label), dev 2,570, test 3,658 (human labels).

Thresholds (shared, then per-language overrides):

- all: agriculture 0.7, finance 0.875, health 0.95, law 0.925, telecoms 0.725
- kin: agriculture 0.7, finance 0.9, health 0.95, law 0.05, telecoms 0.05
- pcm: finance 0.675, health 0.9
- swa: agriculture 0.625, finance 0.775, health 0.3, law 0.625, telecoms 0.05
- yor: agriculture 0.925, finance 0.3, health 0.95, law 0.225, telecoms 0.9

| Language | Test n | Top label correct | Exact label set | Macro F1 | Labels with test data | Human multi-domain | Predicted multi-domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| kin | 385 | 75.1% | 74.5% | 0.712 | 6 of 6 | 0 | 5 |
| pcm | 1533 | 84.9% | 84.9% | 0.780 | 3 of 6 | 0 | 0 |
| swa | 240 | 77.5% | 74.2% | 0.777 | 6 of 6 | 9 | 3 |
| yor | 1500 | 88.5% | 87.7% | 0.896 | 5 of 6 | 0 | 27 |
| ALL | 2158 | 82.3% | 81.8% | 0.753 | 6 of 6 | 9 | 8 |

## kin

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.698 | 0.882 | 0.779 | 68 | 86 |
| finance | 0.717 | 0.600 | 0.653 | 55 | 46 |
| health | 0.705 | 0.662 | 0.683 | 65 | 61 |
| law | 0.833 | 0.983 | 0.902 | 117 | 138 |
| telecoms | 0.889 | 0.615 | 0.727 | 13 | 9 |
| other | 0.620 | 0.463 | 0.530 | 67 | 50 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 60 | 1 | 5 | 0 | 0 | 2 |
| finance | 6 | 33 | 2 | 6 | 0 | 8 |
| health | 9 | 2 | 43 | 3 | 0 | 8 |
| law | 1 | 0 | 0 | 115 | 0 | 1 |
| telecoms | 0 | 5 | 0 | 1 | 7 | 0 |
| other | 10 | 5 | 11 | 9 | 1 | 31 |

## pcm

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.000 | 0.000 | 0.000 | 0 | 14 |
| finance | 0.757 | 0.627 | 0.686 | 169 | 140 |
| health | 0.743 | 0.747 | 0.745 | 170 | 171 |
| law | 0.000 | 0.000 | 0.000 | 0 | 48 |
| telecoms | 0.000 | 0.000 | 0.000 | 0 | 5 |
| other | 0.925 | 0.894 | 0.909 | 1194 | 1155 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 0 | 0 | 0 | 0 | 0 | 0 |
| finance | 0 | 106 | 5 | 1 | 3 | 54 |
| health | 8 | 0 | 127 | 2 | 0 | 33 |
| law | 0 | 0 | 0 | 0 | 0 | 0 |
| telecoms | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 6 | 34 | 39 | 45 | 2 | 1068 |

## swa

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.846 | 0.971 | 0.904 | 34 | 39 |
| finance | 0.773 | 0.944 | 0.850 | 36 | 44 |
| health | 0.897 | 0.921 | 0.909 | 38 | 39 |
| law | 0.750 | 0.789 | 0.769 | 38 | 40 |
| telecoms | 0.562 | 0.818 | 0.667 | 33 | 48 |
| other | 0.879 | 0.414 | 0.563 | 70 | 33 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 33 | 0 | 0 | 0 | 0 | 0 |
| finance | 0 | 34 | 0 | 0 | 1 | 0 |
| health | 1 | 1 | 35 | 0 | 0 | 1 |
| law | 0 | 0 | 0 | 30 | 2 | 3 |
| telecoms | 0 | 4 | 0 | 0 | 25 | 0 |
| other | 5 | 5 | 4 | 10 | 17 | 29 |

## yor

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.900 | 0.927 | 0.913 | 300 | 309 |
| finance | 0.840 | 0.877 | 0.858 | 300 | 313 |
| health | 0.921 | 0.897 | 0.909 | 300 | 292 |
| law | 0.894 | 0.873 | 0.884 | 300 | 293 |
| telecoms | 0.934 | 0.897 | 0.915 | 300 | 288 |
| other | 0.000 | 0.000 | 0.000 | 0 | 32 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 278 | 6 | 8 | 2 | 0 | 6 |
| finance | 8 | 258 | 8 | 9 | 15 | 2 |
| health | 10 | 1 | 269 | 5 | 1 | 14 |
| law | 12 | 22 | 6 | 253 | 3 | 4 |
| telecoms | 1 | 14 | 1 | 9 | 269 | 6 |
| other | 0 | 0 | 0 | 0 | 0 | 0 |

## ALL

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.669 | 0.912 | 0.772 | 102 | 139 |
| finance | 0.752 | 0.665 | 0.706 | 260 | 230 |
| health | 0.756 | 0.751 | 0.754 | 273 | 271 |
| law | 0.642 | 0.935 | 0.761 | 155 | 226 |
| telecoms | 0.565 | 0.761 | 0.648 | 46 | 62 |
| other | 0.911 | 0.847 | 0.878 | 1331 | 1238 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 93 | 1 | 5 | 0 | 0 | 2 |
| finance | 6 | 173 | 7 | 7 | 4 | 62 |
| health | 18 | 3 | 205 | 5 | 0 | 42 |
| law | 1 | 0 | 0 | 145 | 2 | 4 |
| telecoms | 0 | 9 | 0 | 1 | 32 | 0 |
| other | 21 | 44 | 54 | 64 | 20 | 1128 |

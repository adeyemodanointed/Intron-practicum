# AfroXLMR domain classifier: test results

Model `Davlan/afro-xlmr-base`, best checkpoint at epoch 2.50 of 4 epochs (dev mean macro F1 0.815); max length 256, batch 16, lr 2e-05.
Training rows 94,205 (at most 15000 per language and label), dev 2,570, test 3,658 (human labels).

Thresholds (shared, then per-language overrides):

- all: agriculture 0.775, finance 0.95, health 0.925, law 0.875, telecoms 0.925
- kin: agriculture 0.95, finance 0.925, health 0.925, law 0.85, telecoms 0.05
- pcm: finance 0.95, health 0.05
- swa: agriculture 0.05, finance 0.25, health 0.05, law 0.05, telecoms 0.05
- yor: agriculture 0.375, finance 0.375, health 0.15, law 0.775, telecoms 0.925

| Language | Test n | Top label correct | Exact label set | Macro F1 | Labels with test data | Human multi-domain | Predicted multi-domain |
|---|---:|---:|---:|---:|---:|---:|---:|
| kin | 385 | 73.0% | 73.0% | 0.650 | 6 of 6 | 0 | 0 |
| pcm | 1533 | 88.0% | 88.0% | 0.809 | 3 of 6 | 0 | 1 |
| swa | 240 | 76.2% | 72.1% | 0.755 | 6 of 6 | 9 | 5 |
| yor | 1500 | 95.9% | 95.5% | 0.958 | 5 of 6 | 0 | 9 |
| ALL | 2158 | 84.0% | 83.5% | 0.776 | 6 of 6 | 9 | 6 |

## kin

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.766 | 0.868 | 0.814 | 68 | 77 |
| finance | 0.558 | 0.782 | 0.652 | 55 | 77 |
| health | 0.641 | 0.769 | 0.699 | 65 | 78 |
| law | 0.878 | 0.983 | 0.927 | 117 | 131 |
| telecoms | 0.778 | 0.538 | 0.636 | 13 | 9 |
| other | 0.538 | 0.104 | 0.175 | 67 | 13 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 59 | 1 | 5 | 0 | 0 | 3 |
| finance | 3 | 43 | 2 | 5 | 0 | 2 |
| health | 6 | 6 | 50 | 1 | 1 | 1 |
| law | 0 | 0 | 2 | 115 | 0 | 0 |
| telecoms | 0 | 5 | 0 | 1 | 7 | 0 |
| other | 9 | 22 | 19 | 9 | 1 | 7 |

## pcm

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.000 | 0.000 | 0.000 | 0 | 5 |
| finance | 0.796 | 0.669 | 0.727 | 169 | 142 |
| health | 0.763 | 0.776 | 0.770 | 170 | 173 |
| law | 0.000 | 0.000 | 0.000 | 0 | 28 |
| telecoms | 0.000 | 0.000 | 0.000 | 0 | 2 |
| other | 0.932 | 0.925 | 0.929 | 1194 | 1184 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 0 | 0 | 0 | 0 | 0 | 0 |
| finance | 0 | 113 | 4 | 1 | 2 | 49 |
| health | 4 | 1 | 132 | 2 | 0 | 31 |
| law | 0 | 0 | 0 | 0 | 0 | 0 |
| telecoms | 0 | 0 | 0 | 0 | 0 | 0 |
| other | 1 | 28 | 36 | 25 | 0 | 1104 |

## swa

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.805 | 0.971 | 0.880 | 34 | 41 |
| finance | 0.729 | 0.972 | 0.833 | 36 | 48 |
| health | 0.875 | 0.921 | 0.897 | 38 | 40 |
| law | 0.683 | 0.737 | 0.709 | 38 | 41 |
| telecoms | 0.595 | 0.758 | 0.667 | 33 | 42 |
| other | 0.848 | 0.400 | 0.544 | 70 | 33 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 33 | 0 | 0 | 0 | 0 | 0 |
| finance | 0 | 35 | 0 | 0 | 1 | 0 |
| health | 1 | 2 | 34 | 0 | 0 | 1 |
| law | 0 | 1 | 1 | 28 | 1 | 4 |
| telecoms | 0 | 3 | 0 | 0 | 25 | 0 |
| other | 7 | 6 | 2 | 13 | 14 | 28 |

## yor

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.963 | 0.957 | 0.960 | 300 | 298 |
| finance | 0.888 | 0.950 | 0.918 | 300 | 321 |
| health | 0.977 | 0.977 | 0.977 | 300 | 300 |
| law | 0.966 | 0.950 | 0.958 | 300 | 295 |
| telecoms | 0.983 | 0.967 | 0.975 | 300 | 295 |
| other | 0.000 | 0.000 | 0.000 | 0 | 0 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 287 | 8 | 4 | 1 | 0 | 0 |
| finance | 7 | 285 | 0 | 4 | 4 | 0 |
| health | 3 | 0 | 292 | 5 | 0 | 0 |
| law | 0 | 14 | 0 | 285 | 1 | 0 |
| telecoms | 1 | 8 | 1 | 0 | 290 | 0 |
| other | 0 | 0 | 0 | 0 | 0 | 0 |

## ALL

| Label | Precision | Recall | F1 | Test support | Predicted |
|---|---:|---:|---:|---:|---:|
| agriculture | 0.748 | 0.902 | 0.818 | 102 | 123 |
| finance | 0.715 | 0.735 | 0.725 | 260 | 267 |
| health | 0.746 | 0.795 | 0.770 | 273 | 291 |
| law | 0.715 | 0.923 | 0.806 | 155 | 200 |
| telecoms | 0.604 | 0.696 | 0.646 | 46 | 53 |
| other | 0.926 | 0.856 | 0.889 | 1331 | 1230 |

| human \ predicted | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| agriculture | 92 | 1 | 5 | 0 | 0 | 3 |
| finance | 3 | 191 | 6 | 6 | 3 | 51 |
| health | 11 | 9 | 216 | 3 | 1 | 33 |
| law | 0 | 1 | 3 | 143 | 1 | 4 |
| telecoms | 0 | 8 | 0 | 1 | 32 | 0 |
| other | 17 | 56 | 57 | 47 | 15 | 1139 |

# Domain classifier evaluation

Top-label accuracy: the top prediction is one of the human labels. Balanced accuracy: the mean of the per-domain accuracies, so every domain counts equally. Brackets: 95% bootstrap interval. `ALL` = human-labelled texts only.

## Summary

| System | Language | n | Top-label accuracy | Balanced accuracy | Macro F1 | Exact set | Label-set F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| pushed_oct22 | kin | 385 | 71.9% [67.0%–76.4%] | 65.4% [59.5%–71.3%] | 0.674 | 71.7% | 0.719 |
| pushed_oct22 | pcm | 1533 | 86.2% [84.3%–87.7%] | 78.7% [75.5%–81.9%] | 0.793 | 86.2% | 0.862 |
| pushed_oct22 | swa | 240 | 77.5% [72.5%–82.9%] | 79.2% [74.3%–83.7%] | 0.776 | 74.6% | 0.765 |
| pushed_oct22 | ALL | 2158 | 82.7% [81.1%–84.1%] | 79.4% [76.5%–82.2%] | 0.770 | 82.3% | 0.826 |
| base_v8 | kin | 385 | 77.9% [73.8%–81.8%] | 73.3% [67.2%–78.4%] | 0.738 | 77.1% | 0.778 |
| base_v8 | pcm | 1533 | 89.4% [87.8%–91.1%] | 81.7% [78.8%–84.8%] | 0.843 | 89.4% | 0.895 |
| base_v8 | swa | 240 | 80.0% [75.0%–84.6%] | 80.8% [76.3%–85.1%] | 0.795 | 75.8% | 0.786 |
| base_v8 | ALL | 2158 | 86.3% [84.9%–87.7%] | 84.2% [81.5%–86.6%] | 0.799 | 85.7% | 0.862 |
| ensemble_v8 | kin | 385 | 76.1% [71.9%–80.0%] | 72.7% [66.6%–78.0%] | 0.721 | 74.8% | 0.765 |
| ensemble_v8 | pcm | 1533 | 92.0% [90.7%–93.3%] | 86.7% [84.2%–89.4%] | 0.882 | 91.7% | 0.920 |
| ensemble_v8 | swa | 240 | 78.3% [73.3%–83.3%] | 80.7% [75.9%–85.1%] | 0.797 | 75.8% | 0.778 |
| ensemble_v8 | ALL | 2158 | 87.7% [86.2%–89.0%] | 86.2% [83.5%–88.5%] | 0.818 | 86.9% | 0.877 |
| tfidf_v8 | kin | 385 | 74.3% [69.9%–78.4%] | 70.6% [64.8%–76.2%] | 0.699 | 71.9% | 0.742 |
| tfidf_v8 | pcm | 1533 | 91.2% [89.8%–92.5%] | 84.7% [82.2%–87.7%] | 0.862 | 90.7% | 0.910 |
| tfidf_v8 | swa | 240 | 75.8% [70.4%–81.2%] | 77.5% [72.4%–82.1%] | 0.769 | 73.3% | 0.750 |
| tfidf_v8 | ALL | 2158 | 86.5% [84.9%–87.8%] | 83.8% [81.2%–86.3%] | 0.804 | 85.4% | 0.862 |

## Accuracy per domain

Accuracy = of the texts a person gave this domain, the share the model also gave it. Precision = of the texts the model gave this domain, the share that are right. Blank = no texts of that domain in this test set.

### pushed_oct22

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 82.4% (n=68) | 47.3% (n=55) | 61.5% (n=65) | 98.3% (n=117) | 53.8% (n=13) | 49.3% (n=67) |
| pcm |  | 65.1% (n=169) | 81.2% (n=170) |  |  | 89.9% (n=1194) |
| swa | 97.1% (n=34) | 86.1% (n=36) | 92.1% (n=38) | 81.6% (n=38) | 72.7% (n=33) | 45.7% (n=70) |
| ALL | 87.3% (n=102) | 64.2% (n=260) | 78.0% (n=273) | 94.2% (n=155) | 67.4% (n=46) | 85.6% (n=1331) |

### base_v8

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 94.1% (n=68) | 69.1% (n=55) | 72.3% (n=65) | 96.6% (n=117) | 61.5% (n=13) | 46.3% (n=67) |
| pcm |  | 70.4% (n=169) | 81.2% (n=170) |  |  | 93.4% (n=1194) |
| swa | 94.1% (n=34) | 86.1% (n=36) | 94.7% (n=38) | 63.2% (n=38) | 87.9% (n=33) | 58.6% (n=70) |
| ALL | 94.1% (n=102) | 72.3% (n=260) | 81.0% (n=273) | 88.4% (n=155) | 80.4% (n=46) | 89.2% (n=1331) |

### ensemble_v8

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 92.6% (n=68) | 72.7% (n=55) | 69.2% (n=65) | 96.6% (n=117) | 61.5% (n=13) | 43.3% (n=67) |
| pcm |  | 80.5% (n=169) | 84.7% (n=170) |  |  | 94.9% (n=1194) |
| swa | 91.2% (n=34) | 77.8% (n=36) | 92.1% (n=38) | 76.3% (n=38) | 90.9% (n=33) | 55.7% (n=70) |
| ALL | 92.2% (n=102) | 78.5% (n=260) | 82.1% (n=273) | 91.6% (n=155) | 82.6% (n=46) | 90.2% (n=1331) |

### tfidf_v8

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 85.3% (n=68) | 76.4% (n=55) | 67.7% (n=65) | 98.3% (n=117) | 61.5% (n=13) | 34.3% (n=67) |
| pcm |  | 75.1% (n=169) | 84.7% (n=170) |  |  | 94.4% (n=1194) |
| swa | 94.1% (n=34) | 77.8% (n=36) | 92.1% (n=38) | 63.2% (n=38) | 84.8% (n=33) | 52.9% (n=70) |
| ALL | 88.2% (n=102) | 75.8% (n=260) | 81.7% (n=273) | 89.7% (n=155) | 78.3% (n=46) | 89.2% (n=1331) |

## Multi-domain texts

| System | Language | Two-domain texts | Both domains found | Exact set | Single-domain texts | Given a wrong second label |
|---|---|---:|---:|---:|---:|---:|
| pushed_oct22 | kin | 0 | 0 | 0 | 385 | 1 (0.3%) |
| pushed_oct22 | pcm | 0 | 0 | 0 | 1533 | 2 (0.1%) |
| pushed_oct22 | swa | 9 | 0 | 0 | 231 | 0 (0.0%) |
| pushed_oct22 | ALL | 9 | 0 | 0 | 2149 | 3 (0.1%) |
| base_v8 | kin | 0 | 0 | 0 | 385 | 4 (1.0%) |
| base_v8 | pcm | 0 | 0 | 0 | 1533 | 3 (0.2%) |
| base_v8 | swa | 9 | 1 | 1 | 231 | 3 (1.3%) |
| base_v8 | ALL | 9 | 1 | 1 | 2149 | 10 (0.5%) |
| ensemble_v8 | kin | 0 | 0 | 0 | 385 | 12 (3.1%) |
| ensemble_v8 | pcm | 0 | 0 | 0 | 1533 | 10 (0.7%) |
| ensemble_v8 | swa | 9 | 3 | 3 | 231 | 2 (0.9%) |
| ensemble_v8 | ALL | 9 | 3 | 3 | 2149 | 24 (1.1%) |
| tfidf_v8 | kin | 0 | 0 | 0 | 385 | 15 (3.9%) |
| tfidf_v8 | pcm | 0 | 0 | 0 | 1533 | 9 (0.6%) |
| tfidf_v8 | swa | 9 | 2 | 2 | 231 | 1 (0.4%) |
| tfidf_v8 | ALL | 9 | 2 | 2 | 2149 | 25 (1.2%) |

## Paired comparison against pushed_oct22

Difference = system minus baseline on the same texts; p from the paired bootstrap (two-sided); McNemar = exact test on top-label correctness.

| System | Language | Top-label accuracy diff (p) | Balanced accuracy diff (p) | Macro F1 diff (p) | McNemar p |
|---|---|---:|---:|---:|---:|
| base_v8 | kin | +6.0% (0.002) | +7.9% (0.000) | +0.064 (0.006) | 0.002 |
| base_v8 | pcm | +3.3% (0.000) | +2.9% (0.008) | +0.050 (0.000) | 0.000 |
| base_v8 | swa | +2.5% (0.346) | +1.5% (0.466) | +0.019 (0.376) | 0.377 |
| base_v8 | ALL | +3.7% (0.000) | +4.8% (0.000) | +0.030 (0.000) | 0.000 |
| ensemble_v8 | kin | +4.2% (0.012) | +7.2% (0.004) | +0.047 (0.054) | 0.029 |
| ensemble_v8 | pcm | +5.9% (0.000) | +8.0% (0.000) | +0.089 (0.000) | 0.000 |
| ensemble_v8 | swa | +0.8% (0.810) | +1.5% (0.480) | +0.021 (0.300) | 0.856 |
| ensemble_v8 | ALL | +5.0% (0.000) | +6.7% (0.000) | +0.049 (0.000) | 0.000 |
| tfidf_v8 | kin | +2.3% (0.210) | +5.2% (0.064) | +0.026 (0.306) | 0.281 |
| tfidf_v8 | pcm | +5.0% (0.000) | +6.0% (0.000) | +0.069 (0.000) | 0.000 |
| tfidf_v8 | swa | -1.7% (0.502) | -1.7% (0.390) | -0.007 (0.738) | 0.585 |
| tfidf_v8 | ALL | +3.8% (0.000) | +4.4% (0.000) | +0.035 (0.000) | 0.000 |

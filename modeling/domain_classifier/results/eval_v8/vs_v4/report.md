# Domain classifier evaluation

Top-label accuracy: the top prediction is one of the human labels. Balanced accuracy: the mean of the per-domain accuracies, so every domain counts equally. Brackets: 95% bootstrap interval. `ALL` = human-labelled texts only.

## Summary

| System | Language | n | Top-label accuracy | Balanced accuracy | Macro F1 | Exact set | Label-set F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| base_v4_mix | kin | 574 | 74.4% [70.9%–77.9%] | 68.4% [63.9%–73.4%] | 0.674 | 73.9% | 0.747 |
| base_v4_mix | pcm | 2200 | 91.2% [90.0%–92.4%] | 85.3% [82.6%–87.8%] | 0.859 | 91.0% | 0.912 |
| base_v4_mix | swa | 454 | 85.2% [81.7%–88.3%] | 85.2% [82.3%–88.1%] | 0.844 | 81.5% | 0.843 |
| base_v4_mix | yor | 388 | 92.3% [89.4%–94.9%] | 85.4% [80.6%–90.6%] | 0.881 | 92.3% | 0.923 |
| base_v4_mix | ALL | 3616 | 87.9% [86.8%–88.9%] | 86.5% [84.5%–88.1%] | 0.828 | 87.3% | 0.878 |
| base_v8 | kin | 574 | 78.2% [74.7%–81.4%] | 71.5% [66.7%–76.4%] | 0.726 | 77.7% | 0.782 |
| base_v8 | pcm | 2200 | 89.9% [88.6%–91.2%] | 81.2% [78.3%–83.9%] | 0.842 | 89.9% | 0.899 |
| base_v8 | swa | 454 | 83.5% [80.0%–86.8%] | 83.4% [80.1%–86.4%] | 0.825 | 79.3% | 0.824 |
| base_v8 | yor | 388 | 89.2% [86.1%–92.3%] | 82.1% [77.2%–87.4%] | 0.865 | 89.2% | 0.892 |
| base_v8 | ALL | 3616 | 87.2% [86.1%–88.2%] | 84.3% [82.4%–86.0%] | 0.811 | 86.5% | 0.870 |
| tfidf6 | kin | 574 | 72.8% [69.3%–76.5%] | 65.2% [60.8%–70.1%] | 0.646 | 70.9% | 0.724 |
| tfidf6 | pcm | 2200 | 91.2% [90.0%–92.4%] | 85.8% [83.3%–88.2%] | 0.867 | 90.7% | 0.910 |
| tfidf6 | swa | 454 | 79.3% [75.8%–83.0%] | 80.1% [76.8%–83.3%] | 0.786 | 74.7% | 0.785 |
| tfidf6 | yor | 388 | 92.5% [89.7%–95.1%] | 86.0% [81.3%–90.9%] | 0.885 | 92.5% | 0.925 |
| tfidf6 | ALL | 3616 | 86.9% [85.8%–87.9%] | 83.9% [81.8%–85.9%] | 0.809 | 85.7% | 0.867 |

## Accuracy per domain

Accuracy = of the texts a person gave this domain, the share the model also gave it. Precision = of the texts the model gave this domain, the share that are right. Blank = no texts of that domain in this test set.

### base_v4_mix

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 91.2% (n=102) | 76.8% (n=82) | 73.2% (n=97) | 97.2% (n=176) | 47.4% (n=19) | 24.5% (n=98) |
| pcm |  | 77.7% (n=215) | 84.5% (n=239) |  |  | 93.8% (n=1746) |
| swa | 97.2% (n=71) | 88.6% (n=70) | 94.7% (n=75) | 82.2% (n=73) | 89.5% (n=76) | 59.0% (n=105) |
| yor |  |  | 73.8% (n=80) |  |  | 97.1% (n=308) |
| ALL | 93.6% (n=173) | 79.6% (n=367) | 82.1% (n=491) | 92.8% (n=249) | 81.1% (n=95) | 89.6% (n=2257) |

### base_v8

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 92.2% (n=102) | 73.2% (n=82) | 69.1% (n=97) | 97.2% (n=176) | 47.4% (n=19) | 50.0% (n=98) |
| pcm |  | 70.7% (n=215) | 79.1% (n=239) |  |  | 93.8% (n=1746) |
| swa | 94.4% (n=71) | 87.1% (n=70) | 97.3% (n=75) | 71.2% (n=73) | 88.2% (n=76) | 61.9% (n=105) |
| yor |  |  | 70.0% (n=80) |  |  | 94.2% (n=308) |
| ALL | 93.1% (n=173) | 74.4% (n=367) | 78.4% (n=491) | 89.6% (n=249) | 80.0% (n=95) | 90.5% (n=2257) |

### tfidf6

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 90.2% (n=102) | 79.3% (n=82) | 66.0% (n=97) | 96.6% (n=176) | 36.8% (n=19) | 22.4% (n=98) |
| pcm |  | 77.7% (n=215) | 86.2% (n=239) |  |  | 93.6% (n=1746) |
| swa | 94.4% (n=71) | 81.4% (n=70) | 96.0% (n=75) | 75.3% (n=73) | 80.3% (n=76) | 53.3% (n=105) |
| yor |  |  | 75.0% (n=80) |  |  | 97.1% (n=308) |
| ALL | 91.9% (n=173) | 78.7% (n=367) | 81.9% (n=491) | 90.4% (n=249) | 71.6% (n=95) | 89.1% (n=2257) |

## Multi-domain texts

| System | Language | Two-domain texts | Both domains found | Exact set | Single-domain texts | Given a wrong second label |
|---|---|---:|---:|---:|---:|---:|
| base_v4_mix | kin | 0 | 0 | 0 | 574 | 8 (1.4%) |
| base_v4_mix | pcm | 0 | 0 | 0 | 2200 | 7 (0.3%) |
| base_v4_mix | swa | 16 | 3 | 3 | 438 | 7 (1.6%) |
| base_v4_mix | ALL | 16 | 3 | 3 | 3600 | 22 (0.6%) |
| base_v8 | kin | 0 | 0 | 0 | 574 | 6 (1.0%) |
| base_v8 | pcm | 0 | 0 | 0 | 2200 | 3 (0.1%) |
| base_v8 | swa | 16 | 4 | 4 | 438 | 11 (2.5%) |
| base_v8 | yor | 0 | 0 | 0 | 388 | 1 (0.3%) |
| base_v8 | ALL | 16 | 4 | 4 | 3600 | 21 (0.6%) |
| tfidf6 | kin | 0 | 0 | 0 | 574 | 14 (2.4%) |
| tfidf6 | pcm | 0 | 0 | 0 | 2200 | 15 (0.7%) |
| tfidf6 | swa | 16 | 3 | 3 | 438 | 16 (3.7%) |
| tfidf6 | ALL | 16 | 3 | 3 | 3600 | 45 (1.2%) |

## Paired comparison against base_v4_mix

Difference = system minus baseline on the same texts; p from the paired bootstrap (two-sided); McNemar = exact test on top-label correctness.

| System | Language | Top-label accuracy diff (p) | Balanced accuracy diff (p) | Macro F1 diff (p) | McNemar p |
|---|---|---:|---:|---:|---:|
| base_v8 | kin | +3.8% (0.010) | +3.1% (0.080) | +0.052 (0.008) | 0.006 |
| base_v8 | pcm | -1.3% (0.004) | -4.1% (0.000) | -0.017 (0.032) | 0.009 |
| base_v8 | swa | -1.8% (0.132) | -1.8% (0.092) | -0.019 (0.096) | 0.152 |
| base_v8 | yor | -3.1% (0.024) | -3.3% (0.084) | -0.016 (0.378) | 0.029 |
| base_v8 | ALL | -0.7% (0.080) | -2.1% (0.000) | -0.016 (0.004) | 0.090 |
| tfidf6 | kin | -1.6% (0.314) | -3.1% (0.090) | -0.029 (0.170) | 0.328 |
| tfidf6 | pcm | +0.0% (0.996) | +0.5% (0.634) | +0.008 (0.404) | 1.000 |
| tfidf6 | swa | -5.9% (0.000) | -5.1% (0.002) | -0.058 (0.000) | 0.000 |
| tfidf6 | yor | +0.3% (0.898) | +0.6% (0.810) | +0.004 (0.832) | 1.000 |
| tfidf6 | ALL | -1.0% (0.028) | -2.5% (0.000) | -0.019 (0.006) | 0.045 |

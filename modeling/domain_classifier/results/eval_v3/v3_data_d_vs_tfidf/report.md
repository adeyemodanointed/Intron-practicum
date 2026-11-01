# Domain classifier evaluation

Top-label accuracy: the top prediction is one of the human labels. Balanced accuracy: the mean of the per-domain accuracies, so every domain counts equally. Brackets: 95% bootstrap interval. `ALL` = human-labelled texts only.

## Summary

| System | Language | n | Top-label accuracy | Balanced accuracy | Macro F1 | Exact set | Label-set F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| tfidf6_full | kin | 574 | 72.8% [69.3%–76.5%] | 65.2% [60.8%–70.1%] | 0.646 | 70.9% | 0.724 |
| tfidf6_full | pcm | 2200 | 91.2% [90.0%–92.4%] | 85.8% [83.3%–88.2%] | 0.867 | 90.7% | 0.910 |
| tfidf6_full | swa | 454 | 79.3% [75.8%–83.0%] | 80.1% [76.8%–83.3%] | 0.786 | 74.7% | 0.785 |
| tfidf6_full | yor | 388 | 92.5% [89.7%–95.1%] | 86.0% [81.3%–90.9%] | 0.885 | 92.5% | 0.925 |
| tfidf6_full | yor (synthetic) | 3000 | 94.5% [93.7%–95.3%] | 95.9% [95.2%–96.6%] | 0.950 | 91.7% | 0.945 |
| tfidf6_full | ALL | 3616 | 86.9% [85.8%–88.1%] | 83.9% [82.0%–85.9%] | 0.809 | 85.7% | 0.867 |
| base_d_plus_news_data | kin | 574 | 75.1% [71.6%–78.6%] | 68.5% [63.9%–73.3%] | 0.672 | 74.6% | 0.754 |
| base_d_plus_news_data | pcm | 2200 | 90.4% [89.0%–91.6%] | 84.5% [81.9%–86.8%] | 0.858 | 90.3% | 0.905 |
| base_d_plus_news_data | swa | 454 | 83.0% [79.7%–86.1%] | 82.9% [79.8%–86.1%] | 0.820 | 79.3% | 0.819 |
| base_d_plus_news_data | yor | 388 | 92.0% [89.2%–94.6%] | 87.6% [83.4%–92.1%] | 0.885 | 92.0% | 0.920 |
| base_d_plus_news_data | yor (synthetic) | 3000 | 94.8% [94.0%–95.7%] | 95.5% [94.7%–96.2%] | 0.949 | 94.1% | 0.950 |
| base_d_plus_news_data | ALL | 3616 | 87.2% [86.1%–88.3%] | 85.4% [83.5%–87.3%] | 0.814 | 86.6% | 0.872 |

## Accuracy per domain

Accuracy = of the texts a person gave this domain, the share the model also gave it. Precision = of the texts the model gave this domain, the share that are right. Blank = no texts of that domain in this test set.

### tfidf6_full

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 90.2% (n=102) | 79.3% (n=82) | 66.0% (n=97) | 96.6% (n=176) | 36.8% (n=19) | 22.4% (n=98) |
| pcm |  | 77.7% (n=215) | 86.2% (n=239) |  |  | 93.6% (n=1746) |
| swa | 94.4% (n=71) | 81.4% (n=70) | 96.0% (n=75) | 75.3% (n=73) | 80.3% (n=76) | 53.3% (n=105) |
| yor |  |  | 75.0% (n=80) |  |  | 97.1% (n=308) |
| yor (synthetic) | 97.2% (n=600) | 91.0% (n=600) | 97.5% (n=600) | 96.7% (n=600) | 97.3% (n=600) |  |
| ALL | 91.9% (n=173) | 78.7% (n=367) | 81.9% (n=491) | 90.4% (n=249) | 71.6% (n=95) | 89.1% (n=2257) |

### base_d_plus_news_data

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 92.2% (n=102) | 81.7% (n=82) | 72.2% (n=97) | 97.2% (n=176) | 42.1% (n=19) | 25.5% (n=98) |
| pcm |  | 76.7% (n=215) | 83.7% (n=239) |  |  | 93.2% (n=1746) |
| swa | 97.2% (n=71) | 90.0% (n=70) | 94.7% (n=75) | 75.3% (n=73) | 84.2% (n=76) | 56.2% (n=105) |
| yor |  |  | 80.0% (n=80) |  |  | 95.1% (n=308) |
| yor (synthetic) | 97.2% (n=600) | 90.0% (n=600) | 97.2% (n=600) | 96.3% (n=600) | 96.7% (n=600) |  |
| ALL | 94.2% (n=173) | 80.4% (n=367) | 82.5% (n=491) | 90.8% (n=249) | 75.8% (n=95) | 88.8% (n=2257) |

## Multi-domain texts

| System | Language | Two-domain texts | Both domains found | Exact set | Single-domain texts | Given a wrong second label |
|---|---|---:|---:|---:|---:|---:|
| tfidf6_full | kin | 0 | 0 | 0 | 574 | 14 (2.4%) |
| tfidf6_full | pcm | 0 | 0 | 0 | 2200 | 15 (0.7%) |
| tfidf6_full | swa | 16 | 3 | 3 | 438 | 16 (3.7%) |
| tfidf6_full | yor (synthetic) | 0 | 0 | 0 | 3000 | 128 (4.3%) |
| tfidf6_full | ALL | 16 | 3 | 3 | 3600 | 45 (1.2%) |
| base_d_plus_news_data | kin | 0 | 0 | 0 | 574 | 9 (1.6%) |
| base_d_plus_news_data | pcm | 0 | 0 | 0 | 2200 | 6 (0.3%) |
| base_d_plus_news_data | swa | 16 | 3 | 3 | 438 | 6 (1.4%) |
| base_d_plus_news_data | yor (synthetic) | 0 | 0 | 0 | 3000 | 45 (1.5%) |
| base_d_plus_news_data | ALL | 16 | 3 | 3 | 3600 | 21 (0.6%) |

## Paired comparison against tfidf6_full

Difference = system minus baseline on the same texts; p from the paired bootstrap (two-sided); McNemar = exact test on top-label correctness.

| System | Language | Top-label accuracy diff (p) | Balanced accuracy diff (p) | Macro F1 diff (p) | McNemar p |
|---|---|---:|---:|---:|---:|
| base_d_plus_news_data | kin | +2.3% (0.110) | +3.2% (0.038) | +0.027 (0.176) | 0.148 |
| base_d_plus_news_data | pcm | -0.8% (0.130) | -1.3% (0.240) | -0.008 (0.356) | 0.168 |
| base_d_plus_news_data | swa | +3.7% (0.026) | +2.8% (0.084) | +0.034 (0.026) | 0.027 |
| base_d_plus_news_data | yor | -0.5% (0.780) | +1.5% (0.448) | +0.000 (0.950) | 0.839 |
| base_d_plus_news_data | yor (synthetic) | +0.3% (0.504) | -0.5% (0.240) | -0.001 (0.812) | 0.536 |
| base_d_plus_news_data | ALL | +0.3% (0.548) | +1.5% (0.064) | +0.006 (0.428) | 0.602 |

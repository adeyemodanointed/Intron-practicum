# Domain classifier evaluation

Top-label accuracy: the top prediction is one of the human labels. Balanced accuracy: the mean of the per-domain accuracies, so every domain counts equally. Brackets: 95% bootstrap interval. `ALL` = human-labelled texts only.

## Summary

| System | Language | n | Top-label accuracy | Balanced accuracy | Macro F1 | Exact set | Label-set F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| tfidf6_full | kin | 574 | 72.8% [69.3%–76.5%] | 65.2% [60.8%–70.1%] | 0.646 | 70.9% | 0.724 |
| tfidf6_full | pcm | 2200 | 91.2% [90.0%–92.4%] | 85.8% [83.3%–88.2%] | 0.867 | 90.7% | 0.910 |
| tfidf6_full | swa | 454 | 79.3% [75.8%–83.0%] | 80.1% [76.8%–83.3%] | 0.786 | 74.7% | 0.785 |
| tfidf6_full | yor (synthetic) | 3000 | 94.5% [93.7%–95.3%] | 95.9% [95.2%–96.6%] | 0.950 | 91.7% | 0.945 |
| tfidf6_full | ALL | 3228 | 86.2% [85.0%–87.4%] | 83.9% [81.7%–85.9%] | 0.808 | 84.9% | 0.860 |
| tfidf_c_gold | kin | 574 | 73.5% [70.0%–77.2%] | 65.9% [61.4%–70.6%] | 0.644 | 72.1% | 0.734 |
| tfidf_c_gold | pcm | 2200 | 90.9% [89.6%–92.1%] | 87.4% [84.9%–89.5%] | 0.862 | 90.4% | 0.908 |
| tfidf_c_gold | swa | 454 | 79.1% [75.6%–82.8%] | 79.7% [76.4%–82.9%] | 0.786 | 75.1% | 0.783 |
| tfidf_c_gold | yor (synthetic) | 3000 | 95.3% [94.6%–96.1%] | 96.3% [95.6%–96.9%] | 0.960 | 93.5% | 0.953 |
| tfidf_c_gold | ALL | 3228 | 86.1% [84.9%–87.2%] | 84.6% [82.4%–86.4%] | 0.812 | 85.0% | 0.859 |
| base_b_plus_human_labels | kin | 574 | 75.4% [72.1%–78.9%] | 68.5% [63.7%–73.3%] | 0.690 | 75.3% | 0.755 |
| base_b_plus_human_labels | pcm | 2200 | 90.8% [89.5%–92.0%] | 86.0% [83.4%–88.4%] | 0.861 | 90.8% | 0.908 |
| base_b_plus_human_labels | swa | 454 | 80.4% [76.9%–83.7%] | 81.2% [78.1%–84.2%] | 0.791 | 75.8% | 0.794 |
| base_b_plus_human_labels | yor (synthetic) | 3000 | 95.5% [94.8%–96.2%] | 96.7% [96.0%–97.3%] | 0.959 | 94.8% | 0.960 |
| base_b_plus_human_labels | ALL | 3228 | 86.6% [85.5%–87.9%] | 85.7% [83.8%–87.5%] | 0.812 | 85.9% | 0.865 |
| base_c_plus_other_output | kin | 574 | 76.1% [72.6%–79.8%] | 68.7% [64.1%–73.6%] | 0.689 | 75.6% | 0.760 |
| base_c_plus_other_output | pcm | 2200 | 90.4% [89.1%–91.5%] | 84.4% [81.8%–86.7%] | 0.853 | 90.1% | 0.903 |
| base_c_plus_other_output | swa | 454 | 81.5% [78.0%–84.8%] | 82.4% [79.3%–85.4%] | 0.804 | 76.4% | 0.805 |
| base_c_plus_other_output | yor (synthetic) | 3000 | 94.7% [93.9%–95.5%] | 95.1% [94.4%–95.9%] | 0.949 | 94.0% | 0.947 |
| base_c_plus_other_output | ALL | 3228 | 86.6% [85.4%–87.8%] | 84.9% [82.8%–86.9%] | 0.810 | 85.6% | 0.864 |
| base_d_plus_news_data | kin | 574 | 75.1% [71.6%–78.6%] | 68.5% [63.9%–73.3%] | 0.672 | 74.6% | 0.754 |
| base_d_plus_news_data | pcm | 2200 | 90.4% [89.0%–91.6%] | 84.5% [81.9%–86.8%] | 0.858 | 90.3% | 0.905 |
| base_d_plus_news_data | swa | 454 | 83.0% [79.7%–86.1%] | 82.9% [79.8%–86.1%] | 0.820 | 79.3% | 0.819 |
| base_d_plus_news_data | yor (synthetic) | 3000 | 94.8% [94.0%–95.6%] | 95.5% [94.7%–96.2%] | 0.949 | 94.1% | 0.950 |
| base_d_plus_news_data | ALL | 3228 | 86.6% [85.5%–87.8%] | 85.3% [83.4%–87.2%] | 0.815 | 86.0% | 0.866 |

## Accuracy per domain

Accuracy = of the texts a person gave this domain, the share the model also gave it. Precision = of the texts the model gave this domain, the share that are right. Blank = no texts of that domain in this test set.

### tfidf6_full

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 90.2% (n=102) | 79.3% (n=82) | 66.0% (n=97) | 96.6% (n=176) | 36.8% (n=19) | 22.4% (n=98) |
| pcm |  | 77.7% (n=215) | 86.2% (n=239) |  |  | 93.6% (n=1746) |
| swa | 94.4% (n=71) | 81.4% (n=70) | 96.0% (n=75) | 75.3% (n=73) | 80.3% (n=76) | 53.3% (n=105) |
| yor (synthetic) | 97.2% (n=600) | 91.0% (n=600) | 97.5% (n=600) | 96.7% (n=600) | 97.3% (n=600) |  |
| ALL | 91.9% (n=173) | 78.7% (n=367) | 83.2% (n=411) | 90.4% (n=249) | 71.6% (n=95) | 87.8% (n=1949) |

### tfidf_c_gold

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 90.2% (n=102) | 79.3% (n=82) | 73.2% (n=97) | 98.3% (n=176) | 36.8% (n=19) | 17.3% (n=98) |
| pcm |  | 80.9% (n=215) | 88.7% (n=239) |  |  | 92.5% (n=1746) |
| swa | 93.0% (n=71) | 80.0% (n=70) | 94.7% (n=75) | 78.1% (n=73) | 78.9% (n=76) | 53.3% (n=105) |
| yor (synthetic) | 97.5% (n=600) | 92.0% (n=600) | 98.5% (n=600) | 96.2% (n=600) | 97.2% (n=600) |  |
| ALL | 91.3% (n=173) | 80.4% (n=367) | 86.1% (n=411) | 92.4% (n=249) | 70.5% (n=95) | 86.6% (n=1949) |

### base_b_plus_human_labels

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 93.1% (n=102) | 69.5% (n=82) | 70.1% (n=97) | 98.3% (n=176) | 47.4% (n=19) | 32.7% (n=98) |
| pcm |  | 76.7% (n=215) | 88.3% (n=239) |  |  | 92.9% (n=1746) |
| swa | 97.2% (n=71) | 87.1% (n=70) | 93.3% (n=75) | 80.8% (n=73) | 84.2% (n=76) | 44.8% (n=105) |
| yor (synthetic) | 97.2% (n=600) | 92.8% (n=600) | 98.5% (n=600) | 96.8% (n=600) | 98.0% (n=600) |  |
| ALL | 94.8% (n=173) | 77.1% (n=367) | 84.9% (n=411) | 93.2% (n=249) | 76.8% (n=95) | 87.3% (n=1949) |

### base_c_plus_other_output

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 88.2% (n=102) | 79.3% (n=82) | 68.0% (n=97) | 97.7% (n=176) | 42.1% (n=19) | 36.7% (n=98) |
| pcm |  | 80.0% (n=215) | 79.9% (n=239) |  |  | 93.2% (n=1746) |
| swa | 98.6% (n=71) | 88.6% (n=70) | 93.3% (n=75) | 78.1% (n=73) | 84.2% (n=76) | 51.4% (n=105) |
| yor (synthetic) | 97.0% (n=600) | 88.2% (n=600) | 97.8% (n=600) | 95.3% (n=600) | 97.3% (n=600) |  |
| ALL | 92.5% (n=173) | 81.5% (n=367) | 79.6% (n=411) | 92.0% (n=249) | 75.8% (n=95) | 88.1% (n=1949) |

### base_d_plus_news_data

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 92.2% (n=102) | 81.7% (n=82) | 72.2% (n=97) | 97.2% (n=176) | 42.1% (n=19) | 25.5% (n=98) |
| pcm |  | 76.7% (n=215) | 83.7% (n=239) |  |  | 93.2% (n=1746) |
| swa | 97.2% (n=71) | 90.0% (n=70) | 94.7% (n=75) | 75.3% (n=73) | 84.2% (n=76) | 56.2% (n=105) |
| yor (synthetic) | 97.2% (n=600) | 90.0% (n=600) | 97.2% (n=600) | 96.3% (n=600) | 96.7% (n=600) |  |
| ALL | 94.2% (n=173) | 80.4% (n=367) | 83.0% (n=411) | 90.8% (n=249) | 75.8% (n=95) | 87.8% (n=1949) |

## Multi-domain texts

| System | Language | Two-domain texts | Both domains found | Exact set | Single-domain texts | Given a wrong second label |
|---|---|---:|---:|---:|---:|---:|
| tfidf6_full | kin | 0 | 0 | 0 | 574 | 14 (2.4%) |
| tfidf6_full | pcm | 0 | 0 | 0 | 2200 | 15 (0.7%) |
| tfidf6_full | swa | 16 | 3 | 3 | 438 | 16 (3.7%) |
| tfidf6_full | yor (synthetic) | 0 | 0 | 0 | 3000 | 128 (4.3%) |
| tfidf6_full | ALL | 16 | 3 | 3 | 3212 | 45 (1.4%) |
| tfidf_c_gold | kin | 0 | 0 | 0 | 574 | 17 (3.0%) |
| tfidf_c_gold | pcm | 0 | 0 | 0 | 2200 | 13 (0.6%) |
| tfidf_c_gold | swa | 16 | 3 | 3 | 438 | 13 (3.0%) |
| tfidf_c_gold | yor (synthetic) | 0 | 0 | 0 | 3000 | 83 (2.8%) |
| tfidf_c_gold | ALL | 16 | 3 | 3 | 3212 | 43 (1.3%) |
| base_b_plus_human_labels | kin | 0 | 0 | 0 | 574 | 2 (0.3%) |
| base_b_plus_human_labels | pcm | 0 | 0 | 0 | 2200 | 1 (0.0%) |
| base_b_plus_human_labels | swa | 16 | 1 | 1 | 438 | 12 (2.7%) |
| base_b_plus_human_labels | yor (synthetic) | 0 | 0 | 0 | 3000 | 58 (1.9%) |
| base_b_plus_human_labels | ALL | 16 | 1 | 1 | 3212 | 15 (0.5%) |
| base_c_plus_other_output | kin | 0 | 0 | 0 | 574 | 3 (0.5%) |
| base_c_plus_other_output | pcm | 0 | 0 | 0 | 2200 | 7 (0.3%) |
| base_c_plus_other_output | swa | 16 | 2 | 2 | 438 | 15 (3.4%) |
| base_c_plus_other_output | yor (synthetic) | 0 | 0 | 0 | 3000 | 36 (1.2%) |
| base_c_plus_other_output | ALL | 16 | 2 | 2 | 3212 | 25 (0.8%) |
| base_d_plus_news_data | kin | 0 | 0 | 0 | 574 | 9 (1.6%) |
| base_d_plus_news_data | pcm | 0 | 0 | 0 | 2200 | 6 (0.3%) |
| base_d_plus_news_data | swa | 16 | 3 | 3 | 438 | 6 (1.4%) |
| base_d_plus_news_data | yor (synthetic) | 0 | 0 | 0 | 3000 | 45 (1.5%) |
| base_d_plus_news_data | ALL | 16 | 3 | 3 | 3212 | 21 (0.7%) |

## Paired comparison against tfidf6_full

Difference = system minus baseline on the same texts; p from the paired bootstrap (two-sided); McNemar = exact test on top-label correctness.

| System | Language | Top-label accuracy diff (p) | Balanced accuracy diff (p) | Macro F1 diff (p) | McNemar p |
|---|---|---:|---:|---:|---:|
| tfidf_c_gold | kin | +0.7% (0.460) | +0.6% (0.382) | -0.002 (0.844) | 0.523 |
| tfidf_c_gold | pcm | -0.3% (0.360) | +1.6% (0.000) | -0.005 (0.322) | 0.410 |
| tfidf_c_gold | swa | -0.2% (0.924) | -0.5% (0.624) | -0.000 (0.992) | 1.000 |
| tfidf_c_gold | yor (synthetic) | +0.8% (0.004) | +0.3% (0.220) | +0.009 (0.000) | 0.005 |
| tfidf_c_gold | ALL | -0.1% (0.796) | +0.6% (0.200) | +0.004 (0.400) | 0.771 |
| base_b_plus_human_labels | kin | +2.6% (0.062) | +3.3% (0.088) | +0.045 (0.030) | 0.091 |
| base_b_plus_human_labels | pcm | -0.4% (0.546) | +0.2% (0.870) | -0.006 (0.584) | 0.591 |
| base_b_plus_human_labels | swa | +1.1% (0.528) | +1.1% (0.458) | +0.005 (0.728) | 0.583 |
| base_b_plus_human_labels | yor (synthetic) | +1.0% (0.020) | +0.7% (0.044) | +0.009 (0.020) | 0.032 |
| base_b_plus_human_labels | ALL | +0.4% (0.442) | +1.7% (0.042) | +0.004 (0.566) | 0.520 |
| base_c_plus_other_output | kin | +3.3% (0.022) | +3.5% (0.024) | +0.043 (0.016) | 0.027 |
| base_c_plus_other_output | pcm | -0.8% (0.166) | -1.5% (0.194) | -0.014 (0.156) | 0.192 |
| base_c_plus_other_output | swa | +2.2% (0.110) | +2.2% (0.100) | +0.018 (0.170) | 0.154 |
| base_c_plus_other_output | yor (synthetic) | +0.1% (0.772) | -0.8% (0.080) | -0.001 (0.768) | 0.828 |
| base_c_plus_other_output | ALL | +0.3% (0.510) | +1.0% (0.200) | +0.001 (0.786) | 0.548 |
| base_d_plus_news_data | kin | +2.3% (0.110) | +3.2% (0.038) | +0.027 (0.176) | 0.148 |
| base_d_plus_news_data | pcm | -0.8% (0.130) | -1.3% (0.240) | -0.008 (0.356) | 0.168 |
| base_d_plus_news_data | swa | +3.7% (0.026) | +2.8% (0.084) | +0.034 (0.026) | 0.027 |
| base_d_plus_news_data | yor (synthetic) | +0.3% (0.502) | -0.5% (0.312) | -0.001 (0.820) | 0.536 |
| base_d_plus_news_data | ALL | +0.4% (0.462) | +1.4% (0.094) | +0.006 (0.328) | 0.506 |

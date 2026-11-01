# Domain classifier evaluation

Top-label accuracy: the top prediction is one of the human labels. Balanced accuracy: the mean of the per-domain accuracies, so every domain counts equally. Brackets: 95% bootstrap interval. `ALL` = human-labelled texts only.

## Summary

| System | Language | n | Top-label accuracy | Balanced accuracy | Macro F1 | Exact set | Label-set F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| tfidf_v2_setup | kin | 574 | 71.1% [67.6%–74.7%] | 63.3% [59.2%–67.9%] | 0.614 | 69.5% | 0.707 |
| tfidf_v2_setup | pcm | 2200 | 86.4% [84.9%–87.7%] | 76.4% [73.5%–79.1%] | 0.780 | 86.0% | 0.862 |
| tfidf_v2_setup | swa | 454 | 78.4% [74.7%–81.9%] | 79.8% [76.4%–82.8%] | 0.783 | 74.7% | 0.779 |
| tfidf_v2_setup | yor (synthetic) | 3000 | 95.0% [94.2%–95.8%] | 96.2% [95.5%–96.8%] | 0.953 | 92.3% | 0.949 |
| tfidf_v2_setup | ALL | 3228 | 82.5% [81.1%–83.9%] | 80.9% [78.5%–82.9%] | 0.778 | 81.4% | 0.823 |
| tfidf_c_gold | kin | 574 | 73.5% [70.0%–77.2%] | 65.9% [61.4%–70.6%] | 0.644 | 72.1% | 0.734 |
| tfidf_c_gold | pcm | 2200 | 90.9% [89.6%–92.1%] | 87.4% [84.9%–89.5%] | 0.862 | 90.4% | 0.908 |
| tfidf_c_gold | swa | 454 | 79.1% [75.6%–82.8%] | 79.7% [76.4%–82.9%] | 0.786 | 75.1% | 0.783 |
| tfidf_c_gold | yor (synthetic) | 3000 | 95.3% [94.6%–96.1%] | 96.3% [95.6%–96.9%] | 0.960 | 93.5% | 0.953 |
| tfidf_c_gold | ALL | 3228 | 86.1% [84.9%–87.2%] | 84.6% [82.4%–86.4%] | 0.812 | 85.0% | 0.859 |
| base_a_v2_setup | kin | 574 | 72.5% [69.2%–76.1%] | 65.2% [60.8%–69.9%] | 0.622 | 72.5% | 0.726 |
| base_a_v2_setup | pcm | 2200 | 87.2% [85.6%–88.5%] | 80.6% [77.9%–83.3%] | 0.821 | 86.9% | 0.871 |
| base_a_v2_setup | swa | 454 | 77.5% [74.0%–81.3%] | 78.2% [75.2%–81.5%] | 0.769 | 73.8% | 0.769 |
| base_a_v2_setup | yor (synthetic) | 3000 | 95.1% [94.4%–95.8%] | 95.7% [95.0%–96.4%] | 0.951 | 94.2% | 0.952 |
| base_a_v2_setup | ALL | 3228 | 83.2% [82.0%–84.5%] | 82.7% [80.6%–84.6%] | 0.777 | 82.5% | 0.831 |
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
| base_e_plus_multi_domain | kin | 574 | 72.6% [69.2%–76.1%] | 65.5% [60.7%–70.4%] | 0.646 | 71.8% | 0.726 |
| base_e_plus_multi_domain | pcm | 2200 | 90.1% [88.8%–91.4%] | 84.2% [81.5%–86.7%] | 0.852 | 90.0% | 0.901 |
| base_e_plus_multi_domain | swa | 454 | 82.6% [79.3%–85.7%] | 83.2% [80.4%–86.3%] | 0.807 | 76.9% | 0.809 |
| base_e_plus_multi_domain | yor (synthetic) | 3000 | 94.4% [93.6%–95.2%] | 95.0% [94.3%–95.7%] | 0.947 | 93.6% | 0.945 |
| base_e_plus_multi_domain | ALL | 3228 | 86.0% [84.7%–87.2%] | 84.9% [82.9%–86.8%] | 0.802 | 84.9% | 0.857 |
| large_final | kin | 574 | 76.8% [73.5%–80.1%] | 69.6% [65.1%–74.5%] | 0.693 | 76.8% | 0.769 |
| large_final | pcm | 2200 | 90.4% [89.0%–91.7%] | 84.0% [81.3%–86.4%] | 0.859 | 90.2% | 0.905 |
| large_final | swa | 454 | 82.8% [79.5%–86.1%] | 83.4% [80.6%–86.3%] | 0.823 | 78.9% | 0.821 |
| large_final | yor (synthetic) | 3000 | 96.5% [95.8%–97.1%] | 97.1% [96.5%–97.7%] | 0.966 | 95.7% | 0.966 |
| large_final | ALL | 3228 | 86.9% [85.9%–88.1%] | 85.6% [83.8%–87.4%] | 0.819 | 86.2% | 0.869 |

## Accuracy per domain

Accuracy = of the texts a person gave this domain, the share the model also gave it. Precision = of the texts the model gave this domain, the share that are right. Blank = no texts of that domain in this test set.

### tfidf_v2_setup

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 84.3% (n=102) | 76.8% (n=82) | 75.3% (n=97) | 97.2% (n=176) | 36.8% (n=19) | 9.2% (n=98) |
| pcm |  | 64.7% (n=215) | 73.6% (n=239) |  |  | 90.8% (n=1746) |
| swa | 93.0% (n=71) | 82.9% (n=70) | 94.7% (n=75) | 78.1% (n=73) | 81.6% (n=76) | 48.6% (n=105) |
| yor (synthetic) | 97.8% (n=600) | 93.0% (n=600) | 97.7% (n=600) | 95.5% (n=600) | 96.8% (n=600) |  |
| ALL | 87.9% (n=173) | 70.8% (n=367) | 77.9% (n=411) | 91.6% (n=249) | 72.6% (n=95) | 84.4% (n=1949) |

### tfidf_c_gold

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 90.2% (n=102) | 79.3% (n=82) | 73.2% (n=97) | 98.3% (n=176) | 36.8% (n=19) | 17.3% (n=98) |
| pcm |  | 80.9% (n=215) | 88.7% (n=239) |  |  | 92.5% (n=1746) |
| swa | 93.0% (n=71) | 80.0% (n=70) | 94.7% (n=75) | 78.1% (n=73) | 78.9% (n=76) | 53.3% (n=105) |
| yor (synthetic) | 97.5% (n=600) | 92.0% (n=600) | 98.5% (n=600) | 96.2% (n=600) | 97.2% (n=600) |  |
| ALL | 91.3% (n=173) | 80.4% (n=367) | 86.1% (n=411) | 92.4% (n=249) | 70.5% (n=95) | 86.6% (n=1949) |

### base_a_v2_setup

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 92.2% (n=102) | 79.3% (n=82) | 73.2% (n=97) | 98.3% (n=176) | 42.1% (n=19) | 6.1% (n=98) |
| pcm |  | 72.1% (n=215) | 79.5% (n=239) |  |  | 90.1% (n=1746) |
| swa | 94.4% (n=71) | 92.9% (n=70) | 90.7% (n=75) | 67.1% (n=73) | 80.3% (n=76) | 43.8% (n=105) |
| yor (synthetic) | 95.8% (n=600) | 92.8% (n=600) | 98.3% (n=600) | 95.8% (n=600) | 95.8% (n=600) |  |
| ALL | 93.1% (n=173) | 77.7% (n=367) | 80.0% (n=411) | 89.2% (n=249) | 72.6% (n=95) | 83.4% (n=1949) |

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

### base_e_plus_multi_domain

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 88.2% (n=102) | 70.7% (n=82) | 72.2% (n=97) | 96.6% (n=176) | 42.1% (n=19) | 23.5% (n=98) |
| pcm |  | 78.1% (n=215) | 81.6% (n=239) |  |  | 92.9% (n=1746) |
| swa | 97.2% (n=71) | 88.6% (n=70) | 96.0% (n=75) | 74.0% (n=73) | 89.5% (n=76) | 54.3% (n=105) |
| yor (synthetic) | 96.5% (n=600) | 91.5% (n=600) | 95.8% (n=600) | 95.3% (n=600) | 95.8% (n=600) |  |
| ALL | 91.9% (n=173) | 78.5% (n=367) | 82.0% (n=411) | 90.0% (n=249) | 80.0% (n=95) | 87.3% (n=1949) |

### large_final

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 90.2% (n=102) | 84.1% (n=82) | 76.3% (n=97) | 98.3% (n=176) | 42.1% (n=19) | 26.5% (n=98) |
| pcm |  | 78.1% (n=215) | 80.3% (n=239) |  |  | 93.5% (n=1746) |
| swa | 98.6% (n=71) | 91.4% (n=70) | 97.3% (n=75) | 71.2% (n=73) | 85.5% (n=76) | 56.2% (n=105) |
| yor (synthetic) | 98.0% (n=600) | 93.7% (n=600) | 99.3% (n=600) | 97.0% (n=600) | 97.7% (n=600) |  |
| ALL | 93.6% (n=173) | 82.0% (n=367) | 82.5% (n=411) | 90.4% (n=249) | 76.8% (n=95) | 88.1% (n=1949) |

## Multi-domain texts

| System | Language | Two-domain texts | Both domains found | Exact set | Single-domain texts | Given a wrong second label |
|---|---|---:|---:|---:|---:|---:|
| tfidf_v2_setup | kin | 0 | 0 | 0 | 574 | 13 (2.3%) |
| tfidf_v2_setup | pcm | 0 | 0 | 0 | 2200 | 16 (0.7%) |
| tfidf_v2_setup | swa | 16 | 4 | 4 | 438 | 13 (3.0%) |
| tfidf_v2_setup | yor (synthetic) | 0 | 0 | 0 | 3000 | 117 (3.9%) |
| tfidf_v2_setup | ALL | 16 | 4 | 4 | 3212 | 42 (1.3%) |
| tfidf_c_gold | kin | 0 | 0 | 0 | 574 | 17 (3.0%) |
| tfidf_c_gold | pcm | 0 | 0 | 0 | 2200 | 13 (0.6%) |
| tfidf_c_gold | swa | 16 | 3 | 3 | 438 | 13 (3.0%) |
| tfidf_c_gold | yor (synthetic) | 0 | 0 | 0 | 3000 | 83 (2.8%) |
| tfidf_c_gold | ALL | 16 | 3 | 3 | 3212 | 43 (1.3%) |
| base_a_v2_setup | kin | 0 | 0 | 0 | 574 | 2 (0.3%) |
| base_a_v2_setup | pcm | 0 | 0 | 0 | 2200 | 12 (0.5%) |
| base_a_v2_setup | swa | 16 | 0 | 0 | 438 | 6 (1.4%) |
| base_a_v2_setup | yor (synthetic) | 0 | 0 | 0 | 3000 | 48 (1.6%) |
| base_a_v2_setup | ALL | 16 | 0 | 0 | 3212 | 20 (0.6%) |
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
| base_e_plus_multi_domain | kin | 0 | 0 | 0 | 574 | 8 (1.4%) |
| base_e_plus_multi_domain | pcm | 0 | 0 | 0 | 2200 | 8 (0.4%) |
| base_e_plus_multi_domain | swa | 16 | 5 | 5 | 438 | 20 (4.6%) |
| base_e_plus_multi_domain | yor (synthetic) | 0 | 0 | 0 | 3000 | 42 (1.4%) |
| base_e_plus_multi_domain | ALL | 16 | 5 | 5 | 3212 | 36 (1.1%) |
| large_final | kin | 0 | 0 | 0 | 574 | 1 (0.2%) |
| large_final | pcm | 0 | 0 | 0 | 2200 | 13 (0.6%) |
| large_final | swa | 16 | 3 | 3 | 438 | 9 (2.1%) |
| large_final | yor (synthetic) | 0 | 0 | 0 | 3000 | 44 (1.5%) |
| large_final | ALL | 16 | 3 | 3 | 3212 | 23 (0.7%) |

## Paired comparison against tfidf_v2_setup

Difference = system minus baseline on the same texts; p from the paired bootstrap (two-sided); McNemar = exact test on top-label correctness.

| System | Language | Top-label accuracy diff (p) | Balanced accuracy diff (p) | Macro F1 diff (p) | McNemar p |
|---|---|---:|---:|---:|---:|
| tfidf_c_gold | kin | +2.4% (0.008) | +2.6% (0.000) | +0.030 (0.010) | 0.013 |
| tfidf_c_gold | pcm | +4.5% (0.000) | +11.0% (0.000) | +0.082 (0.000) | 0.000 |
| tfidf_c_gold | swa | +0.7% (0.588) | -0.1% (0.946) | +0.004 (0.682) | 0.690 |
| tfidf_c_gold | yor (synthetic) | +0.3% (0.182) | +0.1% (0.700) | +0.007 (0.002) | 0.203 |
| tfidf_c_gold | ALL | +3.6% (0.000) | +3.7% (0.000) | +0.034 (0.000) | 0.000 |
| base_a_v2_setup | kin | +1.4% (0.256) | +1.9% (0.176) | +0.008 (0.654) | 0.268 |
| base_a_v2_setup | pcm | +0.8% (0.120) | +4.2% (0.000) | +0.041 (0.000) | 0.153 |
| base_a_v2_setup | swa | -0.9% (0.632) | -1.6% (0.282) | -0.014 (0.358) | 0.665 |
| base_a_v2_setup | yor (synthetic) | +0.1% (0.922) | -0.4% (0.272) | -0.002 (0.482) | 0.939 |
| base_a_v2_setup | ALL | +0.7% (0.152) | +1.8% (0.016) | -0.001 (0.926) | 0.166 |
| base_b_plus_human_labels | kin | +4.4% (0.002) | +5.2% (0.002) | +0.076 (0.002) | 0.004 |
| base_b_plus_human_labels | pcm | +4.5% (0.000) | +9.6% (0.000) | +0.081 (0.000) | 0.000 |
| base_b_plus_human_labels | swa | +2.0% (0.208) | +1.5% (0.292) | +0.009 (0.524) | 0.272 |
| base_b_plus_human_labels | yor (synthetic) | +0.5% (0.290) | +0.5% (0.166) | +0.006 (0.092) | 0.276 |
| base_b_plus_human_labels | ALL | +4.1% (0.000) | +4.8% (0.000) | +0.034 (0.000) | 0.000 |
| base_c_plus_other_output | kin | +5.1% (0.000) | +5.4% (0.000) | +0.074 (0.000) | 0.000 |
| base_c_plus_other_output | pcm | +4.0% (0.000) | +8.0% (0.000) | +0.073 (0.000) | 0.000 |
| base_c_plus_other_output | swa | +3.1% (0.032) | +2.6% (0.048) | +0.021 (0.094) | 0.049 |
| base_c_plus_other_output | yor (synthetic) | -0.3% (0.518) | -1.0% (0.008) | -0.004 (0.306) | 0.493 |
| base_c_plus_other_output | ALL | +4.1% (0.000) | +4.0% (0.000) | +0.032 (0.000) | 0.000 |
| base_d_plus_news_data | kin | +4.0% (0.002) | +5.2% (0.000) | +0.058 (0.004) | 0.008 |
| base_d_plus_news_data | pcm | +4.0% (0.000) | +8.2% (0.000) | +0.078 (0.000) | 0.000 |
| base_d_plus_news_data | swa | +4.6% (0.006) | +3.1% (0.044) | +0.038 (0.014) | 0.006 |
| base_d_plus_news_data | yor (synthetic) | -0.2% (0.722) | -0.7% (0.074) | -0.003 (0.326) | 0.748 |
| base_d_plus_news_data | ALL | +4.1% (0.000) | +4.5% (0.000) | +0.037 (0.000) | 0.000 |
| base_e_plus_multi_domain | kin | +1.6% (0.282) | +2.3% (0.142) | +0.032 (0.062) | 0.306 |
| base_e_plus_multi_domain | pcm | +3.8% (0.000) | +7.9% (0.000) | +0.072 (0.000) | 0.000 |
| base_e_plus_multi_domain | swa | +4.2% (0.006) | +3.5% (0.024) | +0.024 (0.126) | 0.016 |
| base_e_plus_multi_domain | yor (synthetic) | -0.6% (0.138) | -1.2% (0.004) | -0.006 (0.116) | 0.173 |
| base_e_plus_multi_domain | ALL | +3.4% (0.000) | +4.1% (0.000) | +0.024 (0.000) | 0.000 |
| large_final | kin | +5.7% (0.000) | +6.3% (0.000) | +0.078 (0.000) | 0.000 |
| large_final | pcm | +4.0% (0.000) | +7.6% (0.000) | +0.079 (0.000) | 0.000 |
| large_final | swa | +4.4% (0.010) | +3.6% (0.014) | +0.040 (0.004) | 0.010 |
| large_final | yor (synthetic) | +1.5% (0.000) | +1.0% (0.012) | +0.013 (0.000) | 0.001 |
| large_final | ALL | +4.4% (0.000) | +4.7% (0.000) | +0.042 (0.000) | 0.000 |

# Domain classifier evaluation

Top-label accuracy: the top prediction is one of the human labels. Balanced accuracy: the mean of the per-domain accuracies, so every domain counts equally. Brackets: 95% bootstrap interval. `ALL` = human-labelled texts only.

## Summary

| System | Language | n | Top-label accuracy | Balanced accuracy | Macro F1 | Exact set | Label-set F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| tfidf6_final | kin | 323 | 46.7% [41.2%–52.0%] | 43.0% [38.4%–47.5%] | 0.443 | 46.7% | 0.467 |
| tfidf6_final | pcm | 139 | 57.6% [49.6%–65.5%] | 63.5% [53.8%–72.6%] | 0.545 | 53.2% | 0.561 |
| tfidf6_final | yor | 351 | 60.1% [55.0%–65.5%] | 57.4% [52.3%–63.2%] | 0.598 | 59.5% | 0.601 |
| tfidf6_final | ALL | 813 | 54.4% [51.0%–57.8%] | 50.3% [47.2%–53.9%] | 0.540 | 53.4% | 0.541 |
| base_final_v3m | kin | 323 | 68.7% [63.8%–74.0%] | 70.4% [65.2%–75.6%] | 0.679 | 65.0% | 0.685 |
| base_final_v3m | pcm | 139 | 62.6% [54.7%–71.2%] | 78.4% [69.8%–85.7%] | 0.639 | 61.9% | 0.624 |
| base_final_v3m | yor | 351 | 63.2% [58.4%–68.1%] | 65.4% [60.6%–70.2%] | 0.644 | 63.0% | 0.639 |
| base_final_v3m | ALL | 813 | 65.3% [62.0%–68.5%] | 68.4% [65.1%–71.7%] | 0.662 | 63.6% | 0.655 |
| large_final_v3m | kin | 323 | 63.8% [58.5%–69.0%] | 64.6% [59.7%–69.7%] | 0.631 | 62.8% | 0.641 |
| large_final_v3m | pcm | 139 | 65.5% [56.8%–73.4%] | 80.8% [71.7%–88.2%] | 0.663 | 65.5% | 0.659 |
| large_final_v3m | yor | 351 | 64.4% [59.0%–69.2%] | 66.7% [61.3%–71.1%] | 0.648 | 64.4% | 0.646 |
| large_final_v3m | ALL | 813 | 64.3% [60.8%–67.8%] | 67.0% [63.8%–70.4%] | 0.652 | 64.0% | 0.646 |

## Accuracy per domain

Accuracy = of the texts a person gave this domain, the share the model also gave it. Precision = of the texts the model gave this domain, the share that are right. Blank = no texts of that domain in this test set.

### tfidf6_final

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 37.2% (n=43) | 8.0% (n=50) | 41.5% (n=53) | 50.0% (n=82) | 22.9% (n=35) | 98.4% (n=61) |
| pcm | 66.7% (n=15) | 85.7% (n=7) | 92.3% (n=13) | 61.9% (n=21) | 25.0% (n=8) | 49.3% (n=75) |
| yor | 45.5% (n=44) | 36.7% (n=30) | 90.2% (n=51) | 32.8% (n=64) | 67.3% (n=55) | 72.0% (n=107) |
| ALL | 45.1% (n=102) | 24.1% (n=87) | 68.4% (n=117) | 44.9% (n=167) | 48.0% (n=98) | 71.6% (n=243) |

### base_final_v3m

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 72.1% (n=43) | 48.0% (n=50) | 77.4% (n=53) | 73.2% (n=82) | 82.9% (n=35) | 68.9% (n=61) |
| pcm | 86.7% (n=15) | 85.7% (n=7) | 84.6% (n=13) | 66.7% (n=21) | 100.0% (n=8) | 46.7% (n=75) |
| yor | 79.5% (n=44) | 53.3% (n=30) | 84.3% (n=51) | 29.7% (n=64) | 81.8% (n=55) | 63.6% (n=107) |
| ALL | 77.5% (n=102) | 52.9% (n=87) | 81.2% (n=117) | 55.7% (n=167) | 83.7% (n=98) | 59.7% (n=243) |

### large_final_v3m

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 74.4% (n=43) | 32.0% (n=50) | 71.7% (n=53) | 73.2% (n=82) | 74.3% (n=35) | 62.3% (n=61) |
| pcm | 93.3% (n=15) | 85.7% (n=7) | 84.6% (n=13) | 85.7% (n=21) | 87.5% (n=8) | 48.0% (n=75) |
| yor | 79.5% (n=44) | 56.7% (n=30) | 90.2% (n=51) | 25.0% (n=64) | 89.1% (n=55) | 59.8% (n=107) |
| ALL | 79.4% (n=102) | 44.8% (n=87) | 81.2% (n=117) | 56.3% (n=167) | 83.7% (n=98) | 56.8% (n=243) |

## Multi-domain texts

| System | Language | Two-domain texts | Both domains found | Exact set | Single-domain texts | Given a wrong second label |
|---|---|---:|---:|---:|---:|---:|
| tfidf6_final | kin | 1 | 0 | 0 | 322 | 0 (0.0%) |
| tfidf6_final | pcm | 0 | 0 | 0 | 139 | 7 (5.0%) |
| tfidf6_final | yor | 0 | 0 | 0 | 351 | 3 (0.9%) |
| tfidf6_final | ALL | 1 | 0 | 0 | 812 | 10 (1.2%) |
| base_final_v3m | kin | 1 | 0 | 0 | 322 | 20 (6.2%) |
| base_final_v3m | pcm | 0 | 0 | 0 | 139 | 3 (2.2%) |
| base_final_v3m | yor | 0 | 0 | 0 | 351 | 9 (2.6%) |
| base_final_v3m | ALL | 1 | 0 | 0 | 812 | 32 (3.9%) |
| large_final_v3m | kin | 1 | 1 | 1 | 322 | 7 (2.2%) |
| large_final_v3m | pcm | 0 | 0 | 0 | 139 | 1 (0.7%) |
| large_final_v3m | yor | 0 | 0 | 0 | 351 | 3 (0.9%) |
| large_final_v3m | ALL | 1 | 1 | 1 | 812 | 11 (1.4%) |

## Paired comparison against tfidf6_final

Difference = system minus baseline on the same texts; p from the paired bootstrap (two-sided); McNemar = exact test on top-label correctness.

| System | Language | Top-label accuracy diff (p) | Balanced accuracy diff (p) | Macro F1 diff (p) | McNemar p |
|---|---|---:|---:|---:|---:|
| base_final_v3m | kin | +22.0% (0.000) | +27.4% (0.000) | +0.235 (0.000) | 0.000 |
| base_final_v3m | pcm | +5.0% (0.274) | +14.9% (0.000) | +0.094 (0.070) | 0.337 |
| base_final_v3m | yor | +3.1% (0.332) | +8.0% (0.020) | +0.047 (0.178) | 0.315 |
| base_final_v3m | ALL | +10.9% (0.000) | +18.1% (0.000) | +0.121 (0.000) | 0.000 |
| large_final_v3m | kin | +17.0% (0.000) | +21.7% (0.000) | +0.188 (0.000) | 0.000 |
| large_final_v3m | pcm | +7.9% (0.074) | +17.3% (0.000) | +0.118 (0.022) | 0.090 |
| large_final_v3m | yor | +4.3% (0.162) | +9.3% (0.004) | +0.050 (0.122) | 0.142 |
| large_final_v3m | ALL | +10.0% (0.000) | +16.7% (0.000) | +0.112 (0.000) | 0.000 |

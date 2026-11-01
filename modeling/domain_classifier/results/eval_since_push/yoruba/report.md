# Domain classifier evaluation

Top-label accuracy: the top prediction is one of the human labels. Balanced accuracy: the mean of the per-domain accuracies, so every domain counts equally. Brackets: 95% bootstrap interval. `ALL` = human-labelled texts only.

## Summary

| System | Language | n | Top-label accuracy | Balanced accuracy | Macro F1 | Exact set | Label-set F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| pushed_tfidf | yor | 388 | 69.1% [64.7%–73.5%] | 65.9% [60.4%–71.7%] | 0.741 | 69.1% | 0.692 |
| pushed_tfidf | yor (validated) | 359 | 72.1% [67.7%–76.9%] | 77.4% [73.1%–81.8%] | 0.738 | 71.0% | 0.721 |
| pushed_tfidf | ALL | 747 | 70.5% [67.3%–73.8%] | 77.4% [73.0%–81.2%] | 0.691 | 70.0% | 0.706 |
| pushed_tfidf | ALL (validated) | 359 | 72.1% [67.7%–76.9%] | 77.4% [72.9%–81.8%] | 0.738 | 71.0% | 0.721 |
| base_v8 | yor | 388 | 89.2% [85.6%–92.3%] | 82.1% [76.5%–87.4%] | 0.865 | 89.2% | 0.892 |
| base_v8 | yor (validated) | 359 | 71.6% [67.4%–76.0%] | 71.9% [67.1%–76.7%] | 0.710 | 70.5% | 0.718 |
| base_v8 | ALL | 747 | 80.7% [77.8%–83.5%] | 73.4% [69.0%–77.9%] | 0.729 | 80.2% | 0.808 |
| base_v8 | ALL (validated) | 359 | 71.6% [67.1%–76.1%] | 71.9% [67.0%–76.7%] | 0.710 | 70.5% | 0.718 |
| tfidf_v8 | yor | 388 | 89.2% [85.8%–92.0%] | 84.4% [79.2%–88.9%] | 0.859 | 88.9% | 0.891 |
| tfidf_v8 | yor (validated) | 359 | 73.5% [68.8%–78.3%] | 73.9% [69.4%–78.6%] | 0.732 | 71.3% | 0.739 |
| tfidf_v8 | ALL | 747 | 81.7% [78.7%–84.3%] | 74.4% [69.6%–78.9%] | 0.749 | 80.5% | 0.818 |
| tfidf_v8 | ALL (validated) | 359 | 73.5% [69.1%–78.3%] | 73.9% [69.1%–78.3%] | 0.732 | 71.3% | 0.739 |
| ensemble_v8 | yor | 388 | 89.9% [86.6%–92.5%] | 84.9% [80.0%–89.6%] | 0.873 | 89.2% | 0.897 |
| ensemble_v8 | yor (validated) | 359 | 75.2% [71.0%–79.7%] | 76.1% [71.4%–80.7%] | 0.749 | 73.5% | 0.754 |
| ensemble_v8 | ALL | 747 | 82.9% [80.1%–85.5%] | 77.2% [72.7%–81.7%] | 0.760 | 81.7% | 0.828 |
| ensemble_v8 | ALL (validated) | 359 | 75.2% [71.0%–79.4%] | 76.1% [71.4%–80.4%] | 0.749 | 73.5% | 0.754 |

## Accuracy per domain

Accuracy = of the texts a person gave this domain, the share the model also gave it. Precision = of the texts the model gave this domain, the share that are right. Blank = no texts of that domain in this test set.

### pushed_tfidf

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| yor |  |  | 60.0% (n=80) |  |  | 71.8% (n=308) |
| yor (validated) | 86.4% (n=44) | 83.9% (n=31) | 84.6% (n=52) | 82.8% (n=64) | 75.0% (n=56) | 51.8% (n=112) |
| ALL | 86.4% (n=44) | 83.9% (n=31) | 69.7% (n=132) | 82.8% (n=64) | 75.0% (n=56) | 66.4% (n=420) |
| ALL (validated) | 86.4% (n=44) | 83.9% (n=31) | 84.6% (n=52) | 82.8% (n=64) | 75.0% (n=56) | 51.8% (n=112) |

### base_v8

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| yor |  |  | 70.0% (n=80) |  |  | 94.2% (n=308) |
| yor (validated) | 86.4% (n=44) | 48.4% (n=31) | 84.6% (n=52) | 67.2% (n=64) | 75.0% (n=56) | 69.6% (n=112) |
| ALL | 86.4% (n=44) | 48.4% (n=31) | 75.8% (n=132) | 67.2% (n=64) | 75.0% (n=56) | 87.6% (n=420) |
| ALL (validated) | 86.4% (n=44) | 48.4% (n=31) | 84.6% (n=52) | 67.2% (n=64) | 75.0% (n=56) | 69.6% (n=112) |

### tfidf_v8

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| yor |  |  | 76.2% (n=80) |  |  | 92.5% (n=308) |
| yor (validated) | 81.8% (n=44) | 45.2% (n=31) | 94.2% (n=52) | 70.3% (n=64) | 78.6% (n=56) | 73.2% (n=112) |
| ALL | 81.8% (n=44) | 45.2% (n=31) | 83.3% (n=132) | 70.3% (n=64) | 78.6% (n=56) | 87.4% (n=420) |
| ALL (validated) | 81.8% (n=44) | 45.2% (n=31) | 94.2% (n=52) | 70.3% (n=64) | 78.6% (n=56) | 73.2% (n=112) |

### ensemble_v8

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| yor |  |  | 76.2% (n=80) |  |  | 93.5% (n=308) |
| yor (validated) | 79.5% (n=44) | 58.1% (n=31) | 90.4% (n=52) | 70.3% (n=64) | 85.7% (n=56) | 72.3% (n=112) |
| ALL | 79.5% (n=44) | 58.1% (n=31) | 81.8% (n=132) | 70.3% (n=64) | 85.7% (n=56) | 87.9% (n=420) |
| ALL (validated) | 79.5% (n=44) | 58.1% (n=31) | 90.4% (n=52) | 70.3% (n=64) | 85.7% (n=56) | 72.3% (n=112) |

## Multi-domain texts

| System | Language | Two-domain texts | Both domains found | Exact set | Single-domain texts | Given a wrong second label |
|---|---|---:|---:|---:|---:|---:|
| pushed_tfidf | yor | 0 | 0 | 0 | 388 | 3 (0.8%) |
| pushed_tfidf | yor (validated) | 0 | 0 | 0 | 359 | 7 (1.9%) |
| pushed_tfidf | ALL | 0 | 0 | 0 | 747 | 10 (1.3%) |
| pushed_tfidf | ALL (validated) | 0 | 0 | 0 | 359 | 7 (1.9%) |
| base_v8 | yor | 0 | 0 | 0 | 388 | 1 (0.3%) |
| base_v8 | yor (validated) | 0 | 0 | 0 | 359 | 9 (2.5%) |
| base_v8 | ALL | 0 | 0 | 0 | 747 | 10 (1.3%) |
| base_v8 | ALL (validated) | 0 | 0 | 0 | 359 | 9 (2.5%) |
| tfidf_v8 | yor | 0 | 0 | 0 | 388 | 2 (0.5%) |
| tfidf_v8 | yor (validated) | 0 | 0 | 0 | 359 | 14 (3.9%) |
| tfidf_v8 | ALL | 0 | 0 | 0 | 747 | 16 (2.1%) |
| tfidf_v8 | ALL (validated) | 0 | 0 | 0 | 359 | 14 (3.9%) |
| ensemble_v8 | yor | 0 | 0 | 0 | 388 | 3 (0.8%) |
| ensemble_v8 | yor (validated) | 0 | 0 | 0 | 359 | 10 (2.8%) |
| ensemble_v8 | ALL | 0 | 0 | 0 | 747 | 13 (1.7%) |
| ensemble_v8 | ALL (validated) | 0 | 0 | 0 | 359 | 10 (2.8%) |

## Paired comparison against pushed_tfidf

Difference = system minus baseline on the same texts; p from the paired bootstrap (two-sided); McNemar = exact test on top-label correctness.

| System | Language | Top-label accuracy diff (p) | Balanced accuracy diff (p) | Macro F1 diff (p) | McNemar p |
|---|---|---:|---:|---:|---:|
| base_v8 | yor | +20.1% (0.000) | +16.2% (0.000) | +0.124 (0.000) | 0.000 |
| base_v8 | yor (validated) | -0.6% (0.956) | -5.5% (0.068) | -0.028 (0.316) | 0.924 |
| base_v8 | ALL | +10.2% (0.000) | -4.0% (0.186) | +0.037 (0.096) | 0.000 |
| base_v8 | ALL (validated) | -0.6% (0.854) | -5.5% (0.050) | -0.028 (0.264) | 0.924 |
| tfidf_v8 | yor | +20.1% (0.000) | +18.5% (0.000) | +0.118 (0.000) | 0.000 |
| tfidf_v8 | yor (validated) | +1.4% (0.652) | -3.5% (0.210) | -0.006 (0.806) | 0.685 |
| tfidf_v8 | ALL | +11.1% (0.000) | -2.9% (0.282) | +0.058 (0.012) | 0.000 |
| tfidf_v8 | ALL (validated) | +1.4% (0.656) | -3.5% (0.174) | -0.006 (0.812) | 0.685 |
| ensemble_v8 | yor | +20.9% (0.000) | +19.0% (0.000) | +0.132 (0.000) | 0.000 |
| ensemble_v8 | yor (validated) | +3.1% (0.282) | -1.4% (0.630) | +0.011 (0.634) | 0.305 |
| ensemble_v8 | ALL | +12.3% (0.000) | -0.1% (0.922) | +0.069 (0.000) | 0.000 |
| ensemble_v8 | ALL (validated) | +3.1% (0.288) | -1.4% (0.578) | +0.011 (0.650) | 0.305 |

# Domain classifier evaluation

Top-label accuracy: the top prediction is one of the human labels. Balanced accuracy: the mean of the per-domain accuracies, so every domain counts equally. Brackets: 95% bootstrap interval. `ALL` = human-labelled texts only.

## Summary

| System | Language | n | Top-label accuracy | Balanced accuracy | Macro F1 | Exact set | Label-set F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| base_v8 | kin | 574 | 78.2% [74.7%–81.4%] | 71.5% [66.7%–76.4%] | 0.726 | 77.7% | 0.782 |
| base_v8 | kin (validated) | 356 | 79.2% [75.3%–83.4%] | 77.9% [73.4%–82.5%] | 0.764 | 75.3% | 0.783 |
| base_v8 | pcm | 2258 | 89.7% [88.5%–90.9%] | 84.6% [77.6%–89.6%] | 0.667 | 89.7% | 0.897 |
| base_v8 | pcm (validated) | 327 | 66.4% [60.9%–71.3%] | 76.9% [72.0%–80.9%] | 0.682 | 66.1% | 0.663 |
| base_v8 | swa | 454 | 83.5% [80.2%–87.0%] | 83.4% [80.3%–86.6%] | 0.825 | 79.3% | 0.824 |
| base_v8 | yor | 388 | 89.2% [86.1%–92.3%] | 82.1% [76.8%–87.1%] | 0.865 | 89.2% | 0.892 |
| base_v8 | yor (validated) | 359 | 71.6% [66.9%–76.3%] | 71.9% [66.8%–76.7%] | 0.710 | 70.5% | 0.718 |
| base_v8 | ALL | 4716 | 83.9% [82.8%–84.9%] | 82.6% [81.1%–84.1%] | 0.802 | 83.0% | 0.837 |
| base_v8 | ALL (validated) | 1042 | 72.6% [69.9%–75.2%] | 75.4% [72.7%–78.0%] | 0.726 | 70.7% | 0.723 |

## Accuracy per domain

Accuracy = of the texts a person gave this domain, the share the model also gave it. Precision = of the texts the model gave this domain, the share that are right. Blank = no texts of that domain in this test set.

### base_v8

| Language | agriculture | finance | health | law | telecoms | other |
|---|---|---|---|---|---|---|
| kin | 92.2% (n=102) | 73.2% (n=82) | 69.1% (n=97) | 97.2% (n=176) | 47.4% (n=19) | 50.0% (n=98) |
| kin (validated) | 80.0% (n=45) | 66.0% (n=50) | 76.4% (n=55) | 92.8% (n=111) | 88.6% (n=35) | 63.9% (n=61) |
| pcm |  | 72.0% (n=225) | 79.3% (n=246) | 90.9% (n=11) | 87.5% (n=8) | 93.5% (n=1768) |
| pcm (validated) | 83.8% (n=37) | 90.0% (n=20) | 90.2% (n=41) | 55.6% (n=54) | 100.0% (n=48) | 41.7% (n=127) |
| swa | 94.4% (n=71) | 87.1% (n=70) | 97.3% (n=75) | 71.2% (n=73) | 88.2% (n=76) | 61.9% (n=105) |
| yor |  |  | 70.0% (n=80) |  |  | 94.2% (n=308) |
| yor (validated) | 86.4% (n=44) | 48.4% (n=31) | 84.6% (n=52) | 67.2% (n=64) | 75.0% (n=56) | 69.6% (n=112) |
| ALL | 89.0% (n=299) | 73.0% (n=478) | 79.6% (n=646) | 83.6% (n=489) | 84.3% (n=242) | 86.4% (n=2579) |
| ALL (validated) | 83.3% (n=126) | 65.3% (n=101) | 83.1% (n=148) | 76.9% (n=229) | 87.1% (n=139) | 56.7% (n=300) |

## Multi-domain texts

| System | Language | Two-domain texts | Both domains found | Exact set | Single-domain texts | Given a wrong second label |
|---|---|---:|---:|---:|---:|---:|
| base_v8 | kin | 0 | 0 | 0 | 574 | 6 (1.0%) |
| base_v8 | kin (validated) | 1 | 0 | 0 | 355 | 17 (4.8%) |
| base_v8 | pcm | 0 | 0 | 0 | 2258 | 3 (0.1%) |
| base_v8 | pcm (validated) | 0 | 0 | 0 | 327 | 3 (0.9%) |
| base_v8 | swa | 16 | 4 | 4 | 438 | 11 (2.5%) |
| base_v8 | yor | 0 | 0 | 0 | 388 | 1 (0.3%) |
| base_v8 | yor (validated) | 0 | 0 | 0 | 359 | 9 (2.5%) |
| base_v8 | ALL | 17 | 4 | 4 | 4699 | 50 (1.1%) |
| base_v8 | ALL (validated) | 1 | 0 | 0 | 1041 | 29 (2.8%) |

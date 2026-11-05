# Kinyarwanda Domain Classification — Meeting Snapshot

**Task:** five-way Kinyarwanda text-domain classification: Agriculture,
Finance, Health, Law, and Telecommunications.

## Data status

| Dataset stage | Rows | Detail / status |
|---|---:|---|
| Input corpus | 10,000 | All rows passed preprocessing; no rows were rejected. |
| Model-ready corpus | **10,000** | Exactly 2,000 rows per domain. |
| Train split | 7,000 | 1,400 rows per domain. |
| Validation split | 1,500 | 300 rows per domain; used for learning curves and checkpoint selection. |
| Internal test split | 1,500 | 300 rows per domain; evaluated once for the TF-IDF baseline. |
| Manually reviewed external set | 296 five-domain rows | Derived from 360 reviewed Kinyarwanda rows; `Other`, blank, and multi-label rows were excluded. |

### Training-data provenance

| Domain | Rows | Recorded training source |
|---|---:|---|
| Agriculture | 2,000 | DigitalUmuganda/Afrivoice Kinyarwanda |
| Finance | 2,000 | DigitalUmuganda/Afrivoice Kinyarwanda |
| Health | 2,000 | DigitalUmuganda/Afrivoice Kinyarwanda |
| Law | 2,000 | DigitalUmuganda/Rwandan Gazette parallel data |
| Telecommunications | 2,000 | Synthetic, distributed across 21 telecom sub-topics |
| **Total** | **10,000** | — |

## Validation experiments

### TF-IDF learning curve

TF-IDF uses word and character n-grams with logistic regression. The validation
curve plateaued at roughly 1,050 examples per domain, so the 1,050-per-domain
model was retained as the TF-IDF baseline.

| Train rows per domain | Total train rows | Validation macro-F1 | Validation accuracy |
|---:|---:|---:|---:|
| 350 | 1,750 | 0.9018 | 0.9020 |
| 700 | 3,500 | 0.9166 | 0.9167 |
| **1,050** | **5,250** | **0.9374** | **0.9373** |
| 1,400 | 7,000 | 0.9373 | 0.9373 |

### AfroXLMR + LoRA validation run

| Model | Train rows | Validation rows | Macro-F1 | Accuracy | Status |
|---|---:|---:|---:|---:|---|
| AfroXLMR-base + LoRA | 7,000 | 1,500 | 0.9152 | 0.9153 | Completed |

## Evaluation results

| Model | Evaluation set | Rows | Macro-F1 | Accuracy | Interpretation |
|---|---|---:|---:|---:|---|
| TF-IDF baseline | Internal held-out split | 1,500 | 0.9212 | 0.9213 | Optimistic for real-world use: Telecom was synthetic in both training and this split. |
| TF-IDF baseline | Manual external Kinyarwanda set | 296 | 0.4580 | 0.5777 | External diagnostic result. |
| **AfroXLMR + LoRA** | **Manual external Kinyarwanda set** | **296** | **0.6307** | **0.7061** | Best observed external result. |

### AfroXLMR external per-domain F1

| Domain | F1 | Notes |
|---|---:|---|
| Agriculture | 0.8000 | Strong external performance. |
| Finance | 0.4583 | Frequently confused with Law. |
| Health | 0.7573 | Strong external performance. |
| Law | 0.8046 | Highest recall (0.9375). |
| Telecommunications | 0.3333 | Improved over TF-IDF, but recall remains low (0.2059). |

## Important interpretation

The manual external set exposed a substantial domain shift: TF-IDF fell from
0.9374 validation macro-F1 to 0.4580 on manually reviewed real-world text.
AfroXLMR improved the external result to 0.6307, but Telecom and Finance still
need additional **real, manually confirmed** training data.

Because the same manual external set was used to compare TF-IDF and AfroXLMR,
it should now be treated as an **external diagnostic benchmark**, not as a
strictly untouched final test for further model selection. A fresh reviewed
holdout is required for a publishable final performance claim.

## Evidence and reproducibility

| Item | Location |
|---|---|
| Cleaning/split report | `data/preprocessing/kinyarwanda/kin_preprocessing_report.json` |
| TF-IDF learning curve | `modeling/domain_classification/kinyarwanda/outputs/learning_curve/learning_curve_metrics.csv` |
| TF-IDF internal-test metrics | `evaluation/results/kinyarwanda/tfidf_1050_final/test_metrics.json` |
| TF-IDF external metrics | `evaluation/results/kinyarwanda/tfidf_1050_manual_external/test_metrics.json` |
| AfroXLMR validation metrics | `modeling/domain_classification/kinyarwanda/outputs/afroxlmr_lora/validation_metrics.json` |
| AfroXLMR external metrics | `evaluation/results/kinyarwanda/afroxlmr_lora_manual_external/external_metrics.json` |

## Recommended next work

1. Collect and manually review real Kinyarwanda Telecommunications and Finance
   examples; do not use the existing manual external set for training.
2. Retrain AfroXLMR with the expanded real-data corpus.
3. Evaluate once on a new, untouched manually reviewed Kinyarwanda holdout.

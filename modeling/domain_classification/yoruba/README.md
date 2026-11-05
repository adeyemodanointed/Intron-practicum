# Yoruba Domain Classification — Meeting Snapshot

**Task:** five-way Yoruba text-domain classification: Agriculture, Finance,
Health, Law, and Telecommunications.

## Data status

| Dataset stage | Rows | Detail / status |
|---|---:|---|
| Raw seed corpus | 10,239 | Preserved source file; includes 9,999 Gemini-generated rows and 240 AfriDocMT Health documents. |
| Section-audited synthetic additions | 6,239 | New generated rows are assigned to controlled domain sub-sections (for example, cardiology, SIM registration, irrigation, contracts). |
| Model-ready corpus | **10,000** | After cleaning to 12–60 words, removing 239 overlong AfriDocMT documents. No blank or exact-normalised duplicate text was found. |
| Train split | 7,000 | 1,400 rows per domain. |
| Validation split | 1,500 | 300 rows per domain; used only for checkpoint selection and development metrics. |
| Internal test split | 1,500 | 300 rows per domain; protected and **not yet evaluated**. |
| Manually reviewed external Yoruba set | 247 five-domain rows | Protected source exists in `Practicum - Manual Validation.csv`; it has **not yet been used** for Yoruba model evaluation. |

### Training-data balance

| Domain | Model-ready rows | Train | Validation | Internal test |
|---|---:|---:|---:|---:|
| Agriculture | 2,000 | 1,400 | 300 | 300 |
| Finance | 2,000 | 1,400 | 300 | 300 |
| Health | 2,000 | 1,400 | 300 | 300 |
| Law | 2,000 | 1,400 | 300 | 300 |
| Telecommunications | 2,000 | 1,400 | 300 | 300 |
| **Total** | **10,000** | **7,000** | **1,500** | **1,500** |

## Model runs and results

| Model | Training data | Selection set | Macro-F1 | Accuracy | Status |
|---|---:|---:|---:|---:|---|
| AfroXLMR-base + LoRA | 7,000 | 1,500-row validation set | **0.9484** | **0.9487** | Completed |
| TF-IDF word + character n-grams + logistic regression | — | — | — | — | Config ready; not yet run |
| AfroXLMR-base + LoRA | — | Internal test | — | — | Not yet run |
| AfroXLMR-base + LoRA | — | Manual external Yoruba set | — | — | Not yet run |

### Completed AfroXLMR validation results

| Domain | Precision | Recall | F1 | Validation rows |
|---|---:|---:|---:|---:|
| Agriculture | 0.9600 | 0.9600 | 0.9600 | 300 |
| Finance | 0.9333 | 0.8867 | 0.9094 | 300 |
| Health | 0.9455 | 0.9833 | 0.9641 | 300 |
| Law | 0.9493 | 0.9367 | 0.9430 | 300 |
| Telecommunications | 0.9544 | 0.9767 | 0.9654 | 300 |
| **Macro average** | **0.9485** | **0.9487** | **0.9484** | **1,500** |

## Important interpretation

The completed score is a **validation result on a split of predominantly
synthetic data**, not a final real-world claim. It is useful evidence that the
model learns the constructed task, but the protected internal test and manually
reviewed external Yoruba set are still required to assess generalisation.

## Evidence and reproducibility

| Item | Location |
|---|---|
| Raw generated seed | `data/generation/synthetic_text/yoruba/yoruba_synthetic_seed.csv` |
| Sub-section audit | `data/generation/synthetic_text/yoruba/yoruba_synthetic_seed_sections.csv` |
| Cleaning/split report | `data/preprocessing/yoruba/yoruba_synthetic_preprocessing_report.json` |
| Model configuration | `modeling/domain_classification/configs/yoruba_afroxlmr_lora.yaml` |
| Saved adapter | `modeling/domain_classification/yoruba/outputs/afroxlmr_lora/best_adapter/` |
| Validation metrics | `modeling/domain_classification/yoruba/outputs/afroxlmr_lora/validation_metrics.json` |

## Next experiments

1. Run the TF-IDF learning curve using
   `modeling/domain_classification/configs/yoruba_learning_curve.yaml`.
2. Extract the protected manually reviewed Yoruba five-domain rows and evaluate
   the already-trained AfroXLMR adapter exactly once.
3. Report the external score separately from the synthetic-split validation
   score.

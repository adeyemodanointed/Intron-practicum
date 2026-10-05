# Multilingual PnC: Yoruba pilot

This repository contains the current research and implementation work for a multilingual punctuation-and-capitalisation (PnC) practicum. The immediate pilot language is Yoruba (`yor`); Nigerian Pidgin and other African languages are planned follow-on languages.

The work has two connected tracks:

1. **PnC restoration**: restore commas, sentence-ending punctuation, questions, and eventually capitalisation in unpunctuated transcript-like text.
2. **Data mining and domain routing**: collect, clean, audit, review, and route Yoruba web text into defensible topic/domain candidates.

## Current status

### Yoruba PnC data pipeline

The pipeline ingests local and Hugging Face-derived sources, preserves Yoruba diacritics, normalises Unicode and whitespace, filters malformed/short records, deduplicates text, and creates PnC labels by removing punctuation and casing from the model input while retaining the original text as the target.

The current clean Yoruba PnC core contains **157,805 records** and **14,083,057 word tokens**.

| Audit measure | Result |
| --- | ---: |
| Records with some punctuation | 99.21% |
| Records ending with sentence punctuation | 93.82% |
| Commas per 100 word tokens | 5.51 |
| Periods per 100 word tokens | 3.50 |
| Questions per 100 word tokens | 0.07 |

The core provides enough volume for a first baseline, but it is **not domain balanced**: YANKARI accounts for roughly 99% of current token volume and is tagged with an unknown domain. Health terminology and synthetic finance material are kept outside the natural-text PnC core.

Detailed reports: `reports/yoruba_punctuation_audit.csv` and `reports/yoruba_cleaning_log.csv`.

### PnC baseline result

The current learned baseline fine-tunes `Davlan/afro-xlmr-base` for punctuation boundary classification using four labels: `O`, `COMMA`, `PERIOD`, and `QUESTION`.

| Setting | Value |
| --- | --- |
| Training / development / held-out test records | 5,000 / 500 / 500 |
| Epochs | 3 |
| Maximum sequence length | 96 |
| Batch size | 1 |
| Optimizer | Adafactor |

Held-out punctuation results:

| Label | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Comma | 0.599 | 0.355 | 0.446 |
| Period | 0.750 | 0.816 | 0.782 |
| Question | 0.000 | 0.000 | 0.000 |
| **Punctuation macro F1** |  |  | **0.409** |

The rule-only baseline obtained punctuation macro F1 of 0.000. The learned model improves on sentence-ending punctuation and moderately on commas. Question marks are a rare class (12 test instances), and are the clearest next improvement target.

> Capitalisation is not yet a learned result. The current prediction path applies a simple initial-word capitalisation rule, so the reported capitalisation score must not be interpreted as a model result.

Baseline artefacts are in `outputs/yoruba_afroxlmr_gpu_baseline/evaluation/`.

### Domain-data mining and routing

Candidate sources are converted to one shared JSONL structure, reviewed by source/domain, and retained only when the team accepts their language and domain quality. The currently approved Yoruba seed inventory is:

| Domain | Approved records | Word tokens | 10k natural-token target |
| --- | ---: | ---: | --- |
| General news | 2,747 | 77,496 | Met |
| Education | 241 | 4,926 | Not yet met |
| Religion | 318 | 4,595 | Not yet met |

A provisional document/segment router was trained with character TF-IDF features and balanced logistic regression. It routes `general_news`, `education`, and `religion` candidates and achieved **development macro F1 = 0.894** on a 662-record held-out split.

This is a feasibility result, not final performance: education and religion are below the 10k target, and source-independent evaluation is still required.

Domain-router artefacts:

- `outputs/yoruba_domain_router_pilot/metrics.json`
- `outputs/yoruba_domain_router_pilot/confusion_matrix.csv`
- `outputs/yoruba_domain_router_pilot/model.joblib`

## Data governance

- Raw and formatted data are **not automatically approved for redistribution**. Preserve source URLs, licences, and review decisions.
- Synthetic or translated text may be used only as explicitly tagged augmentation; it does not count toward the natural-domain token target and must not enter final evaluation.
- Gated Intron Health data is not included in the current results. It will be audited for licence, language, text quality, and domain before use.
- Treat model-routed web text as `needs_review`, even at high confidence.

## Repository layout

```text
configs/              Experiment configurations
data_pipeline/        Reproducible Yoruba cleaning and audit pipeline
domain_routing/       Candidate builder, reviewer, router, and Wikipedia scraper
scripts/              PnC preparation, training, prediction, and evaluation
data/                 Local data outputs (not all suitable for publication)
reports/              Audits and data-quality summaries
outputs/              Trained models, predictions, and evaluation artefacts
```

## Reproducing the PnC baseline

Run from the repository root after activating an environment with PyTorch, Transformers, Datasets, scikit-learn, and the project installed.

```powershell
# Build the cleaned Yoruba core and audits
python data_pipeline\yoruba_pipeline.py run

# Prepare supervised PnC records
python scripts\prepare_data.py --config configs\yoruba_gpu_baseline.yaml --input data\processed\yoruba_clean.jsonl --output data\processed\yoruba_supervised.jsonl

# Train with a CUDA-enabled PyTorch environment
python scripts\train.py --config configs\yoruba_gpu_baseline.yaml --train data\processed\yoruba_gpu_baseline\train.jsonl --dev data\processed\yoruba_gpu_baseline\dev.jsonl --output-dir outputs\yoruba_afroxlmr_gpu_baseline --model-name models\afro-xlmr-base

# Predict and evaluate on the held-out test split
python scripts\predict.py --input data\processed\yoruba_gpu_baseline\test.jsonl --output outputs\yoruba_afroxlmr_gpu_baseline\test_predictions.jsonl --model-dir outputs\yoruba_afroxlmr_gpu_baseline\model --config configs\yoruba_gpu_baseline.yaml
python scripts\evaluate.py --predictions outputs\yoruba_afroxlmr_gpu_baseline\test_predictions.jsonl --output-dir outputs\yoruba_afroxlmr_gpu_baseline\evaluation
```

## Domain-routing workflow

```powershell
python domain_routing\build_seed_candidates.py
python domain_routing\review_seed_candidates.py --reviewer YourName
python domain_routing\domain_classifier.py validate-seeds --input data\domain_seeds\yoruba\seed_documents.jsonl --report reports\yoruba_domain_seed_audit.json
python domain_routing\domain_classifier.py train --input data\domain_seeds\yoruba\seed_documents.jsonl --output-dir outputs\yoruba_domain_router_pilot --allow-under-target
python domain_routing\domain_classifier.py route --model outputs\yoruba_domain_router_pilot\model.joblib --input data\raw\yoruba\scraped_candidates.jsonl --output data\formatted\yoruba\routed_candidates.jsonl --threshold 0.80
```

## Next steps

1. Audit and integrate approved Intron Health material as a natural health-domain candidate source.
2. Expand education, religion, health, finance, and telecom to the verified 10,000-token target where feasible.
3. Create source/document-aware train/dev/test splits for both routing and PnC.
4. Add class balancing or focal loss and more interrogative text for PnC question marks.
5. Repeat the same baseline protocol for Nigerian Pidgin, then compare separate models with shared Yoruba-Pidgin multilingual training.

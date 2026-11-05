# Multilingual Domain Classification — Results Summary

**Prepared:** 22 October 2026
**Task:** classify short text into **Agriculture, Finance, Health, Law,** or
**Telecommunications** for Kinyarwanda, Yoruba, and Nigerian Pidgin.

This document is designed as a presentation source. All scores are reported as
**macro-F1** and **accuracy**. Macro-F1 is the primary metric because it gives
every domain equal weight, including smaller classes such as Finance in the
manual Pidgin set.

## Slide-ready headline

> AfroXLMR-LoRA is the recommended model family. It generalized better than
> the TF-IDF baseline on manually reviewed external data for all three
> languages, with the strongest result in Nigerian Pidgin (**0.9484 macro-F1**)
> and Yoruba (**0.8486 macro-F1**). Kinyarwanda remains the key improvement
> area (**0.6307 macro-F1**), especially for Telecommunications and Finance.

## 1. Data used

| Language | Model-ready corpus | Train / validation / internal test | External manually reviewed five-domain set | Training-data note |
|---|---:|---:|---:|---|
| Kinyarwanda | 10,000 | 7,000 / 1,500 / 1,500 | 296 | Telecom training examples are synthetic; other domains combine collected sources. |
| Yoruba | 10,000 | 7,000 / 1,500 / 1,500 | 247 | Predominantly synthetic; 239 overlong imported Health documents were excluded by the 12–60-word rule. |
| Nigerian Pidgin | 500 | 350 / 75 / 75 | 214 | All training rows are synthetic; external set comes from six real-data sources. |

All three model-ready corpora are balanced across the five domains. The manual
sets are **not used for model training**.

## 2. Main external-evaluation results

These are the most meaningful results: the models were evaluated on manually
reviewed text that was separate from the training corpus.

| Language | Model | External rows | Macro-F1 | Accuracy | Result |
|---|---|---:|---:|---:|---|
| Kinyarwanda | TF-IDF + logistic regression | 296 | 0.4580 | 0.5777 | Baseline |
| Kinyarwanda | **AfroXLMR-base + LoRA** | 296 | **0.6307** | **0.7061** | Best observed |
| Yoruba | TF-IDF + logistic regression | 247 | 0.8101 | 0.8057 | Baseline |
| Yoruba | **AfroXLMR-base + LoRA** | 247 | **0.8486** | **0.8583** | Best observed |
| Nigerian Pidgin | TF-IDF + logistic regression | 214 | 0.6508 | 0.6636 | Baseline |
| Nigerian Pidgin | **AfroXLMR-base + LoRA** | 214 | **0.9484** | **0.9486** | Best observed |

### AfroXLMR improvement over TF-IDF on external data

| Language | TF-IDF macro-F1 | AfroXLMR macro-F1 | Difference |
|---|---:|---:|---:|
| Kinyarwanda | 0.4580 | 0.6307 | **+0.1727** |
| Yoruba | 0.8101 | 0.8486 | **+0.0385** |
| Nigerian Pidgin | 0.6508 | 0.9484 | **+0.2977** |

## 3. Validation learning curves

TF-IDF learning curves use a fixed validation split and were used to select the
baseline training size. These figures are **development results**, not the
external generalization results above.

### Kinyarwanda

| Train rows/domain | Total train rows | Validation macro-F1 | Accuracy |
|---:|---:|---:|---:|
| 350 | 1,750 | 0.9018 | 0.9020 |
| 700 | 3,500 | 0.9166 | 0.9167 |
| **1,050** | **5,250** | **0.9374** | **0.9373** |
| 1,400 | 7,000 | 0.9373 | 0.9373 |

**Decision:** plateau at 1,050 examples per domain; retain that TF-IDF
baseline.

### Yoruba

| Train rows/domain | Total train rows | Validation macro-F1 | Accuracy |
|---:|---:|---:|---:|
| 350 | 1,750 | 0.9175 | 0.9180 |
| 700 | 3,500 | 0.9458 | 0.9460 |
| 1,050 | 5,250 | 0.9538 | 0.9540 |
| **1,400** | **7,000** | **0.9592** | **0.9593** |

**Decision:** use 1,400 examples per domain for the TF-IDF baseline.

### Nigerian Pidgin

| Train rows/domain | Total train rows | Validation macro-F1 | Accuracy |
|---:|---:|---:|---:|
| 20 | 100 | 0.8098 | 0.8133 |
| 40 | 200 | 0.7981 | 0.8000 |
| **70** | **350** | **0.8683** | **0.8667** |

**Decision:** the available corpus only supports 70 examples per domain. The
external evaluation shows that AfroXLMR is substantially more robust than this
small-data TF-IDF baseline.

## 4. Internal/synthetic-held-out checks

| Language | Model | Held-out split | Rows | Macro-F1 | Accuracy |
|---|---|---|---:|---:|---:|
| Kinyarwanda | TF-IDF | Internal test | 1,500 | 0.9212 | 0.9213 |
| Yoruba | TF-IDF | Internal test | 1,500 | 0.9541 | 0.9540 |
| Yoruba | AfroXLMR-LoRA | Internal test | 1,500 | 0.9519 | 0.9520 |
| Nigerian Pidgin | AfroXLMR-LoRA | Synthetic validation | 75 | 0.9347 | 0.9333 |

These high internal scores show that models learn the constructed task, but
external manual scores are the more reliable indicator of real-world use.

## 5. Per-language interpretation

### Kinyarwanda — improvement priority

- AfroXLMR improved external macro-F1 from **0.4580** to **0.6307** over TF-IDF.
- Strongest AfroXLMR external F1: Agriculture (0.8000), Law (0.8046), and
  Health (0.7573).
- Main gaps: Finance (0.4583) and Telecommunications (0.3333).
- Recommended work: collect additional real, manually confirmed Finance and
  Telecommunications examples, retrain, then test on a fresh reviewed holdout.

### Yoruba — strong generalization

- AfroXLMR external macro-F1 is **0.8486**, compared with 0.8101 for TF-IDF.
- Strongest external F1: Law (0.9037), Telecommunications (0.8807), and
  Agriculture (0.8736).
- Finance is the lowest AfroXLMR external class F1 (0.7778).
- Recommended model: **AfroXLMR-LoRA**.

### Nigerian Pidgin — strongest completed outcome

- AfroXLMR external macro-F1 is **0.9484**, far above TF-IDF at 0.6508.
- Only 11 of 214 in-domain manually reviewed texts were misclassified.
- AfroXLMR per-domain F1: Agriculture 0.9333, Finance 0.9545, Health 0.9348,
  Law 0.9391, Telecommunications 0.9804.
- Recommended model: **AfroXLMR-LoRA**.

## 6. Methodological note for presentation

The reported manual external results are **closed five-domain classification**
scores. Rows manually labelled `Other` were excluded because the current models
do not include an `Other`/unknown class. This is correct for the current task,
but it does not measure open-set rejection.

The same external manual sets were used to report TF-IDF and AfroXLMR results.
They should now be treated as **external diagnostic benchmarks**, not reused to
choose or tune additional models. A fresh manually reviewed holdout is needed
for a strict final publication claim after further model selection.

## 7. Recommended model choice

| Language | Recommended current model | Reason |
|---|---|---|
| Kinyarwanda | AfroXLMR-LoRA | Best observed external generalization; collect more real Telecom/Finance data next. |
| Yoruba | AfroXLMR-LoRA | Best observed external generalization (0.8486 macro-F1). |
| Nigerian Pidgin | AfroXLMR-LoRA | Large advantage over TF-IDF (0.9484 vs. 0.6508 macro-F1). |

## 8. Reproducibility pointers

| Item | Location |
|---|---|
| Kinyarwanda data split | `data/splits/kinyarwanda/Data.csv.gz` |
| Yoruba data split | `data/splits/yoruba_synthetic/Data.csv.gz` |
| Pidgin data split | `data/splits/pidgin_synthetic/Data.csv.gz` |
| AfroXLMR fine-tuning script | `modeling/domain_classification/finetune_transformer.py` |
| TF-IDF learning-curve script | `modeling/domain_classification/learning_curve.py` |
| LoRA external-evaluation script | `evaluation/scripts/evaluate_lora_classifier.py` |
| TF-IDF external-evaluation script | `evaluation/scripts/evaluate_saved_classifier.py` |

# Multilingual Domain Classification — Consolidated Results Brief

## Scope

This document summarises the completed five-domain text-classification work for
**Yoruba, Kinyarwanda, Nigerian Pidgin, and Swahili**. Each model assigns one
text to exactly one of these classes:

> Agriculture · Finance · Health · Law · Telecommunications

This is a **closed-set** task. It does not yet identify unrelated text as
`Other`; every input receives one of the five labels.

## What was done

The same core workflow was used in every language:

1. Prepare an already-labelled corpus, normalise text, de-duplicate it, and
   create deterministic train/validation/internal-test splits.
2. Train a TF-IDF word-and-character n-gram Logistic Regression baseline.
3. Fine-tune an African multilingual encoder (**AfroXLMR**) using LoRA.
4. Select the preferred model using validation performance.
5. Evaluate the selected model on a protected internal test split and/or a
   separate manually reviewed external set, depending on the completed runs.

Macro-F1 is the main metric because it gives equal importance to each domain;
accuracy is reported alongside it as the overall percentage correct.

## Data overview

| Language | Model-ready corpus | Train / validation / internal test | Manually reviewed in-domain external rows | Important data note |
|---|---:|---|---:|---|
| Kinyarwanda | 10,000 | 7,000 / 1,500 / 1,500 | 296 | Telecom training data are synthetic; other domains combine collected text sources. |
| Yoruba | 10,000 | 7,000 / 1,500 / 1,500 | 247 | Predominantly synthetic; health includes a small AfriDocMT component before length filtering. |
| Nigerian Pidgin | 500 | 350 / 75 / 75 | 214 | All training data are synthetic; manual evaluation uses six real-data sources. |
| Swahili | 9,997 | 6,997 / 1,500 / 1,500 | 254 | Sampled deterministically from the labelled Swahili corpus; one exact overlap was removed from external evaluation. |

All corpora are balanced, or nearly balanced after cleaning, across the five
domains.

## Main results

### External manually reviewed evaluation

These results are the most useful indication of performance beyond the
training splits. The Swahili external score was produced only after AfroXLMR
had been selected on validation; the older external sets were used to compare
TF-IDF and AfroXLMR and are therefore now diagnostic benchmarks rather than
untouched future final tests.

| Language | Model | Rows | Macro-F1 | Accuracy | Interpretation |
|---|---|---:|---:|---:|---|
| Kinyarwanda | TF-IDF | 296 | 0.4580 | 0.5777 | Lexical baseline; weak real-world generalisation. |
| Kinyarwanda | **AfroXLMR-LoRA** | 296 | **0.6307** | **0.7061** | Preferred current model; Finance and Telecom need work. |
| Yoruba | TF-IDF | 247 | 0.8101 | 0.8057 | Strong baseline. |
| Yoruba | **AfroXLMR-LoRA** | 247 | **0.8486** | **0.8583** | Preferred current model. |
| Nigerian Pidgin | TF-IDF | 214 | 0.6508 | 0.6636 | Baseline struggles with small synthetic training set. |
| Nigerian Pidgin | **AfroXLMR-LoRA** | 214 | **0.9484** | **0.9486** | Strongest completed external result. |
| Swahili | TF-IDF | 254 | 0.8840 | 0.8858 | Strong lexical baseline on the reviewed set. |
| Swahili | **AfroXLMR-LoRA** | 254 | **0.9311** | **0.9331** | Best observed; interpret beside the lower internal test. |

### AfroXLMR advantage over TF-IDF on external data

| Language | TF-IDF macro-F1 | AfroXLMR macro-F1 | Difference |
|---|---:|---:|---:|
| Kinyarwanda | 0.4580 | 0.6307 | +0.1727 |
| Yoruba | 0.8101 | 0.8486 | +0.0385 |
| Nigerian Pidgin | 0.6508 | 0.9484 | +0.2977 |
| Swahili | 0.8840 | 0.9311 | +0.0471 |

### Internal / development results

| Language | Model / split | Rows | Macro-F1 | Accuracy | Meaning |
|---|---|---:|---:|---:|---|
| Kinyarwanda | TF-IDF internal test | 1,500 | 0.9212 | 0.9213 | Constructed-split result; more optimistic than external performance. |
| Yoruba | TF-IDF internal test | 1,500 | 0.9541 | 0.9540 | Constructed-split result. |
| Yoruba | AfroXLMR internal test | 1,500 | 0.9519 | 0.9520 | Constructed-split result. |
| Nigerian Pidgin | AfroXLMR synthetic validation | 75 | 0.9347 | 0.9333 | Small synthetic validation set. |
| Swahili | AfroXLMR validation | 1,500 | 0.6480 | 0.6460 | Used to select the model/checkpoint. |
| Swahili | AfroXLMR protected internal test | 1,500 | 0.6566 | 0.6553 | Conservative held-out estimate from the sampled corpus. |

## TF-IDF learning curves

| Language | Training rows per domain tried | Best validation macro-F1 | Decision |
|---|---|---:|---|
| Kinyarwanda | 350 / 700 / 1,050 / 1,400 | 0.9374 at 1,050 | Plateaued; retain 1,050/domain TF-IDF baseline. |
| Yoruba | 350 / 700 / 1,050 / 1,400 | 0.9592 at 1,400 | Use 1,400/domain TF-IDF baseline. |
| Nigerian Pidgin | 20 / 40 / 70 | 0.8683 at 70 | Data limited; use available 70/domain baseline. |
| Swahili | 350 / 700 / 1,050 / 1,399 | 0.5920 at 1,399 | Still improving; more diverse labelled data may help. |

## What the results mean

### 1. AfroXLMR is the recommended model family

AfroXLMR-LoRA achieved the best observed reviewed-data score in every language
where both models were evaluated externally. It is especially valuable where
word overlap alone is not enough, because it can represent contextual and
cross-lingual language patterns.

### 2. Data quality and domain coverage matter more than high internal scores

Kinyarwanda and Yoruba show very high scores on their internal splits but lower
scores on manually reviewed text. This is evidence of domain/source shift—not
necessarily a failed model. The manual results are more relevant for real-world
use than synthetic or same-source held-out splits.

### 3. Current language-level conclusions

| Language | Current conclusion | Most important next improvement |
|---|---|---|
| Kinyarwanda | AfroXLMR is a clear improvement, but 0.6307 external macro-F1 is not yet strong enough for a broad deployment claim. | Add real, reviewed Finance and Telecommunications examples. |
| Yoruba | Strong current generalisation (0.8486 external macro-F1). | Expand independently sourced Finance text and create a fresh holdout for any future tuning. |
| Nigerian Pidgin | Very strong result despite a small synthetic training set. | Validate on a new independently reviewed set before making a broad final claim. |
| Swahili | Promising result: 0.6566 internal test and 0.9311 reviewed external evaluation. | Investigate why the external set is easier; collect a new source-disjoint holdout and more diverse Telecom text. |

### 4. Swahili's large score gap is a caution, not an error by itself

The Swahili external result is high even after removing one exact training-text
match. However, its external sources are related to sources represented in the
training corpus and may contain clearer domain cues than the random internal
sample. Report both Swahili numbers together rather than presenting 0.9311 as a
universal performance estimate.

## Important limitations for presentations and publication

- These are **five-way closed-domain** scores. They do not show whether the
  model can reject unrelated text. To support that, collect labelled `Other`
  examples and retrain/evaluate an open-set version.
- Synthetic or weakly labelled data can be useful for training, but they are
  not evidence of real-world performance on their own.
- The Kinyarwanda, Yoruba, and Pidgin external sets were used to compare model
  variants. Do not reuse their outcomes to tune another model and still call
  them untouched final tests.
- Any future model/data changes should be evaluated on a **new**, manually
  reviewed, source-disjoint holdout set.

## Recommended current model choice

| Language | Recommended model | Evidence |
|---|---|---|
| Kinyarwanda | AfroXLMR-LoRA | Best observed reviewed macro-F1: 0.6307. |
| Yoruba | AfroXLMR-LoRA | Best observed reviewed macro-F1: 0.8486. |
| Nigerian Pidgin | AfroXLMR-LoRA | Best observed reviewed macro-F1: 0.9484. |
| Swahili | AfroXLMR-LoRA | Higher validation macro-F1 than TF-IDF; 0.6566 internal test and 0.9311 reviewed external score. |

## Where to find the evidence

| Item | Location |
|---|---|
| Shared preprocessing | `data/preprocessing/preprocess.py` |
| Balanced sampling (Swahili) | `data/preprocessing/sample_balanced_domains.py` |
| TF-IDF training | `modeling/domain_classification/learning_curve.py` |
| AfroXLMR-LoRA training | `modeling/domain_classification/finetune_transformer.py` |
| Saved evaluation outputs | `evaluation/results/<language>/` |
| Swahili-specific detail | `modeling/domain_classification/swahili/README.md` |
| Methodology literature | `METHODOLOGY_REFERENCES.md` |

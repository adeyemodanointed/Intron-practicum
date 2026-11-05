# Swahili Five-Domain Classification: Results Brief

## Purpose

This experiment classifies Swahili text into one of five closed domains:

- Agriculture
- Finance
- Health
- Law
- Telecommunications

The aim was to establish a reproducible Swahili domain-classification baseline
using the same workflow already applied to Yoruba, Kinyarwanda, and Nigerian
Pidgin: a transparent TF-IDF baseline, followed by AfroXLMR with LoRA
fine-tuning.

## Data used

### Training corpus

`data/swa_domain_corpus.csv.gz` was the labelled source corpus. A deterministic
streaming sampler selected 2,000 rows per domain (10,000 rows total) using
`data/preprocessing/sample_balanced_domains.py`.

Preprocessing removed two texts that were too short and one duplicate. The
model-ready corpus therefore contains **9,997 rows**:

| Domain | Accepted rows | Train | Validation | Protected internal test |
|---|---:|---:|---:|---:|
| Agriculture | 1,999 | 1,399 | 300 | 300 |
| Finance | 1,999 | 1,399 | 300 | 300 |
| Health | 2,000 | 1,400 | 300 | 300 |
| Law | 2,000 | 1,400 | 300 | 300 |
| Telecommunications | 1,999 | 1,399 | 300 | 300 |
| **Total** | **9,997** | **6,997** | **1,500** | **1,500** |

The split is deterministic and stratified. The internal test split was not read
by either training script.

### External manually reviewed evaluation

The Swahili portion of `Practicum - Manual Validation.csv` contains 360 rows.
This project is a closed five-domain classifier, so its 105 `Other` rows are
not included in the five-class score. One exact text match with the sampled
training data was found and removed before evaluation.

The resulting external evaluation set contains **254 manually reviewed,
five-domain rows**:

| Domain | Rows |
|---|---:|
| Agriculture | 50 |
| Finance | 48 |
| Health | 55 |
| Law | 56 |
| Telecommunications | 45 |
| **Total** | **254** |

This set is independent at the exact-text level, but it includes sources that
are related to sources in the training corpus. It should therefore be described
as a **manually reviewed external diagnostic evaluation**, not as an untouched
test set for future model selection.

## Models and model-selection protocol

1. **TF-IDF + Logistic Regression** was trained as the lexical baseline. It
   combines word 1–2 grams and character 3–5 grams.
2. **AfroXLMR + LoRA** was fine-tuned as the multilingual transformer model.
   Only LoRA adapter parameters were trained; the base encoder was
   `../PnC/models/afro-xlmr-base`.
3. Learning-curve/validation results selected AfroXLMR. The selected adapter
   was then evaluated once on the protected internal test and once on the
   external reviewed data. Neither evaluation was used for further tuning.

Macro-F1 is reported alongside accuracy because macro-F1 gives the same weight
to every domain, rather than allowing a more common or easier domain to dominate
the result.

## Results

### TF-IDF validation learning curve

| Training examples per domain | Total train rows | Macro-F1 | Accuracy |
|---:|---:|---:|---:|
| 350 | 1,750 | 0.541 | 0.541 |
| 700 | 3,500 | 0.565 | 0.566 |
| 1,050 | 5,250 | 0.570 | 0.571 |
| 1,399 | 6,995 | 0.592 | 0.593 |

The TF-IDF curve was still improving at the largest available size. It is a
useful baseline, but not the selected final model because AfroXLMR achieved a
higher validation macro-F1.

### External model comparison

| Model | External reviewed rows | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 254 | 0.8840 | 0.8858 |
| **AfroXLMR + LoRA** | **254** | **0.9311** | **0.9331** |

AfroXLMR improves macro-F1 by **0.0471** over the TF-IDF baseline on the same
reviewed rows. This makes the Swahili comparison consistent with the other
languages. The external set is now an external diagnostic benchmark and must
not be used for further model/data tuning.

### AfroXLMR + LoRA detailed results

| Evaluation | Rows | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| Validation during training | 1,500 | 0.6480 | 0.6460 |
| Protected internal test | 1,500 | 0.6566 | 0.6553 |
| Manual external diagnostic evaluation | 254 | 0.9311 | 0.9331 |

External per-domain F1:

| Domain | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Agriculture | 0.980 | 0.980 | 0.980 | 50 |
| Finance | 0.842 | 1.000 | 0.914 | 48 |
| Health | 0.932 | 1.000 | 0.965 | 55 |
| Law | 0.942 | 0.875 | 0.907 | 56 |
| Telecommunications | 1.000 | 0.800 | 0.889 | 45 |

## What the results mean

- **AfroXLMR is the preferred Swahili model.** Its validation macro-F1 (0.648)
  exceeds the strongest TF-IDF validation result (0.592), showing that the
  multilingual transformer captures domain signals beyond words and phrases
  alone.
- **The internal result is the more conservative performance estimate.** The
  0.657 internal-test macro-F1 reflects performance on a random held-out split
  of the sampled corpus.
- **The reviewed external result is very encouraging but needs careful
  interpretation.** Its 0.931 macro-F1 confirms strong performance on the 254
  reviewed rows after exact-overlap removal. However, the large gap from the
  internal test suggests that the external set is easier, has clearer domain
  cues, or is source-related to the training corpus. It should not be presented
  as a universal real-world performance claim.
- **Telecommunications is the weakest external domain.** Its recall is 0.800,
  meaning some genuine telecom text is being assigned to another domain. This
  is the clearest candidate for additional diverse labelled data.

## Recommended next steps

1. Retain AfroXLMR-LoRA as the selected five-domain Swahili model.
2. Review the telecom errors in
   `evaluation/results/swahili/afroxlmr_lora_manual_external/external_predictions.csv`
   and add diverse, independently sourced telecom examples to the training set.
3. For a future final claim, create a new manual test set from sources not used
   for training or current evaluation, and do not inspect its results until the
   model and rules are frozen.
4. If deployment must reject non-target text, create an explicit, independently
   labelled `Other` class. The current model is closed-set: it always returns
   one of the five domains.

## Reproducibility files

- Preprocessing config: `data/preprocessing/configs/swahili.yaml`
- TF-IDF config: `modeling/domain_classification/configs/swahili_learning_curve.yaml`
- AfroXLMR config: `modeling/domain_classification/configs/swahili_afroxlmr_lora.yaml`
- Internal-test config: `evaluation/scripts/configs/swahili_afroxlmr_internal_test.yaml`
- External-evaluation config: `evaluation/scripts/configs/swahili_afroxlmr_manual_external.yaml`
- Sampling script: `data/preprocessing/sample_balanced_domains.py`

# One multilingual domain classifier (20 Oct 2026)

One model for every language. It labels a text with **one or two** of agriculture, finance, health, law, telecoms, or **other** when no domain fits. Languages with training data now: Swahili (`swa`), Kinyarwanda (`kin`), Nigerian Pidgin (`pcm`). Igbo and Yoruba can be added in the same format (see "Adding a language").

All scores are on **human-labelled test sets** that the model never trained on.

## Results

| Language | Test texts | One model: top label correct | One model: macro F1 | Separate per-language model: top label correct | Separate model: macro F1 |
|---|---:|---:|---:|---:|---:|
| Swahili | 240 | 70.8% | 0.726 | 75.4% | 0.765 |
| Kinyarwanda | 385 | 71.2% | 0.659 | 70.1% | 0.636 |
| Pidgin | 1,533 | 87.5% | 0.782 | 85.6% | 0.754 |
| All | 2,158 | 82.7% | 0.764 | – | – |

- **One model vs separate models:** the single model is about equal or better for Kinyarwanda and Pidgin, and 4.6 points lower for Swahili. The 95% margin of error is about ±5.5 points for Swahili, ±4.3 for Kinyarwanda and ±2.2 for Pidgin. So none of these gaps is clearly real except Pidgin's gain.
- **Pidgin's test has only health, finance and other.** Its law, telecoms and agriculture labels are not tested.
- Full per-label tables and confusion matrices are in `results/report.md`, and every prediction is in `results/test_predictions.csv`.

Per-label weak points (one model):

| Language | Weak label | Precision | Recall | Why |
|---|---|---:|---:|---|
| Kinyarwanda | other | 0.35 | 0.22 | No Kinyarwanda `other` text in training; out-of-domain texts get forced into health or finance |
| Swahili | telecoms | 0.50 | 0.88 | BBC-style political news (XL-Sum, MasakhaNEWS) gets labelled telecoms; Swahili `other` training text is Tanzanian sports and entertainment news |
| Swahili | other | 0.56 | 0.43 | Same cause |
| Pidgin | finance | 0.81 | 0.57 | Missed finance texts go to `other` |

## Multi-domain labels

The model can output two domains: every domain whose probability passes its threshold, up to two. All training labels have one domain, though, so it rarely finds the second one. Of the 9 Swahili test texts that a human gave two domains, the top label was one of the human's labels in 7, and both were found in 1. Improving this needs multi-label training examples, such as more human annotations like the Swahili 360 sheet.

## How it works

- **Features:** word 1–2 grams and character 3–5 grams (TF-IDF). Text is lower-cased with diacritics removed, and a language-tag token is added.
- **Model:** one class-balanced logistic regression per domain, trained on all languages together.
- **Sample weights:** human labels 1.0, keyword/source labels 0.5. Each language is rescaled to the same total weight, because Kinyarwanda has 2–5× more rows.
- **Thresholds:** one per domain, tuned on the dev sets. Where a language has at least 5 dev examples of a domain, it gets its own threshold; otherwise the shared one is used (`model/thresholds.json`). Below every threshold → `other`.
- **Leakage control:** a training row is dropped if it matches a dev/test text exactly or shares 3+ word 8-grams with one. That removed 91 Swahili, 797 Kinyarwanda and 267 Pidgin rows.

Versions tried (the test set was looked at for all three, so the chosen numbers are slightly optimistic):

| Version | Swahili | Kinyarwanda | Pidgin | All |
|---|---:|---:|---:|---:|
| v1: one threshold per domain for all languages | 65.0% | 72.2% | 85.5% | 80.8% |
| **v2: per-language thresholds (chosen, simplest)** | 70.8% | 71.2% | 87.5% | 82.7% |
| v3: v2 + per-language feature copies (`--augment`) | 72.5% | 69.9% | 86.6% | 82.0% |

## Data

| Language | Train | Dev (threshold tuning) | Test | Where it comes from |
|---|---:|---:|---:|---|
| Swahili | 20,863 | 214 | 240 | Train: licence-clear `Data.csv`, news tier, plus 4,415 `other` chunks from Swahili news (sport, entertainment, and national/international news with no domain keyword). Dev: 94 review-kept passages + 120 of the 360 human labels. Test: the other 240. |
| Kinyarwanda | 49,809 | 189 | 385 | Train: `kinyarwanda_multidomain_candidates.csv` (source-default labels). Dev/test: the 576-row Gold Test, split 1/3 and 2/3, with `others` → `other`. |
| Pidgin | 19,409 | 667 | 1,533 | `pcm_results.zip` train/dev/test. `education` → `other`. 455 keyword-labelled `other` rows with 2+ agriculture keywords were relabelled agriculture, because the Pidgin pipeline treated agriculture as other. |

`data/` holds corpus text. Keep it out of the public repository, and keep `model/domain_classifier.joblib` out too, because its vocabulary is built from that text.

## Run it

```
pip install scikit-learn pandas openpyxl joblib scipy pyarrow
python build_domain_data.py                    # -> data/domain_{train,dev,test}.jsonl (about 1 minute)
python train_domain_classifier.py --compare    # -> model/, results/ (about 10 minutes on a laptop CPU; 4 without --compare)
python predict_domain.py in.csv out.csv --text-col Text --lang swa
```

`predict_domain.py` adds `domains` (for example `finance|telecoms` or `other`), `primary_domain` and `p_<domain>` columns.

## Adding a language (Igbo, Yoruba, more Kinyarwanda `other`)

Give `build_domain_data.py --extra file.csv`, with columns `language, text, labels, split` (and optionally `source, label_source`).
- `labels` is `finance`, `finance|law` or `other`.
- `split` is `train`, `dev` or `test`.

Human-labelled rows go in dev/test. A language with no training rows is labelled `other` for almost everything; a Yoruba test sentence about farming came out `other`.

## AfroXLMR version (GPU)

`train_afroxlmr_domain.py` fine-tunes `Davlan/afro-xlmr-base` as the same one-model, six-label classifier. It uses the same data files, thresholds and scoring as above, so the results tables compare directly.
- **Outputs:** one sigmoid per domain, at most two domains, `other` when none passes its threshold.
- **Training data:** at most 4,000 rows per language and label (45,685 rows).
- **Epochs:** 3, keeping the epoch with the best dev score.
- **Predicting:** `predict_domain_afroxlmr.py` labels any CSV/JSONL. A language without its own thresholds (Igbo, Yoruba) uses the shared ones, and AfroXLMR was pre-trained on both.

**Results (21 Oct 2026, Colab T4, about 31 minutes):** these are on the same human test sets. The best checkpoint is from epoch 1 of 3.

| Language | AfroXLMR: top label correct | TF-IDF one model | AfroXLMR macro F1 | TF-IDF macro F1 |
|---|---:|---:|---:|---:|
| Swahili (240) | 75.0% | 70.8% | 0.756 | 0.726 |
| Kinyarwanda (385) | 73.5% | 71.2% | 0.674 | 0.659 |
| Pidgin (1,533) | 85.8% | 87.5% | 0.785 | 0.782 |
| All (2,158) | 82.4% | 82.7% | 0.754 | 0.764 |

AfroXLMR is a few points better for Swahili and Kinyarwanda and slightly worse for Pidgin. Every gap is within the test sets' margin of error, so the two models are about even. The larger gains need more data rather than a bigger model.

The trained model, report, metrics and predictions are in Google Drive under `MyDrive/afroxlmr_domain_classifier/`. The model is in `afroxlmr_out/model`; pass that folder as `--model-dir` to `predict_domain_afroxlmr.py`.

Run on a GPU with `afroxlmr_domain_classifier_colab.ipynb`: upload `domain_classifier_kaggle_data.zip` to the Colab Files panel, then run the cells in order. `afroxlmr_domain_classifier_kaggle.ipynb` does the same on Kaggle, which needs a phone-verified account. The zip holds corpus text, so share it only privately.

## What would improve it most

1. **Kinyarwanda `other` training text** (for example sports or politics news, or Kinyarwanda Wikipedia on unrelated topics). This is the largest single error.
2. **Igbo and Yoruba data in the shared format.** The Pidgin/Igbo pipeline also needs to switch from `education` to `agriculture`, to match the five domains in the brief.
3. **Swahili `other` text in the BBC / XL-Sum news style**, so political news stops landing in telecoms and law.
4. **Human test labels for Pidgin law, telecoms and agriculture,** and a fresh untouched test set for each language before quoting final numbers.
5. **A multilingual transformer (`Davlan/afro-xlmr-base`, multi-label) on a Colab GPU.** It can carry what it learns from one language to another, which this TF-IDF model cannot. The same data files work as input.

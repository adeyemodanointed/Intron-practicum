# Multilingual domain classifier

One model labels a text in **Swahili (`swa`), Kinyarwanda (`kin`), Nigerian Pidgin (`pcm`) or Yoruba (`yor`)** with **one or two** of:

**agriculture · finance · health · law · telecoms**, or **other** when none of them fits.

It was built for the CMU-Africa MSIT practicum with Intron, to sort domain text for multilingual QA-pair generation. Igbo was left out by team decision.

## Results (22 October 2026)

Top label correct means the model's first label matches a human label. All test texts were held out from training.

| Language | Test texts | Test labels by | Simple model | AfroXLMR (untuned) | **Final model** | Final macro F1 |
|---|---:|---|---:|---:|---:|---:|
| Swahili | 240 | people | 73.3% | 77.1% | **77.5%** | 0.776 |
| Kinyarwanda | 385 | people | 73.2% | 70.6% | **71.9%** | 0.674 |
| Nigerian Pidgin | 1,533 | people | 86.0% | 86.3% | **86.2%** | 0.793 |
| Yoruba | 1,500 | Gemini (synthetic) | 95.2% | 92.7% | **96.3%** | 0.962 |
| All human-labelled | 2,158 | people | 82.3% | 82.5% | **82.7%** | 0.770 |

- **Final model:** three AfroXLMR runs averaged, then mixed 50/50 with the simple model. The mix was chosen on dev data only.
- **Margin of error** (about 95% confidence): ±5 points for Swahili, ±4.5 for Kinyarwanda, ±2 for Pidgin, ±1.6 for all human-labelled texts. Single training runs also differ by about 2 points, so small gaps between models can be chance.
- **Yoruba:** the test sentences were written by an AI and have no off-topic examples, so real Yoruba accuracy will be lower.
- **Remaining errors** mostly involve `other`. 192 off-topic texts got a domain (often Kinyarwanda), and 138 domain texts were called `other` (mostly Pidgin finance and health news).
- **Two domains:** supported, but rarely predicted, because every training text has a single label. Of 9 human two-domain test texts, the top label was right in 7; both labels were found in none.

The full presentation, with charts, per-label scores, the error table and example predictions, is in `client_notebook/Domain_Classifier_Results.ipynb`.

## How we measure

| Metric | Meaning |
|---|---|
| **Top label correct** (headline) | The model's first label is one of the human labels |
| **Exact label set** | The model's full set of labels equals the human set (stricter) |
| **Precision / recall / F1** (per label) | How often a predicted label is right / how many real cases are found / a balance of the two (1.0 is perfect) |
| **Macro F1** | Average F1 over the labels that have test texts, each label weighted equally, so weak rare labels pull it down |
| **Dev mean macro F1** | Macro F1 on dev, averaged over languages. It chose the learning rate, the best checkpoint and the model mix. The test set was never used for choices |

## How it works

1. **Data** (`build_domain_data.py`): all languages are pooled into one train/dev/test set with the six labels. A training row is dropped if it matches a dev/test text or shares 3+ word 8-grams with one; this removed 2,189 rows.
2. **Simple model** (`train_domain_classifier.py`): word and character TF-IDF, with one class-balanced logistic regression per domain and a language tag. It runs on a laptop CPU in about 5 minutes.
3. **AfroXLMR** (`train_afroxlmr_domain.py`): `Davlan/afro-xlmr-base` fine-tuned with one sigmoid output per domain, on a GPU.
   - **Tuning** (on Kaggle, one overnight run of 4.3 hours on a T4):
     - learning rate 1e-5 / 2e-5 / 3e-5, with 2e-5 chosen;
     - up to 15,000 rows per language and label (about 95,000 rows);
     - up to 4 epochs, with dev checked every half epoch and the best checkpoint kept;
     - 3 seeds.
4. **Final model** (`ensemble_domain.py`): averages the seeds, mixes in the simple model, and tunes the thresholds on dev.
5. **Labelling rule:** a domain is given when its score passes a per-domain threshold, tuned per language where dev has enough examples. A text gets at most two domains, and `other` if none passes.

Training samples are weighted: human labels 1.0, keyword or source labels 0.5. Each language's total weight is equal.

## Data

| Language | Train | Dev | Test | Sources |
|---|---:|---:|---:|---|
| Swahili | 20,863 | 214 | 240 | Licence-checked web, Wikipedia, legal and agricultural documents and news (`Data.csv` and the news tier). `other` comes from sport, entertainment and general news. The test set is 240 hand-labelled passages |
| Kinyarwanda | 49,809 | 189 | 385 | The team's multidomain corpus, with source-default labels. Dev and test come from the team's hand-labelled gold sample |
| Nigerian Pidgin | 19,409 | 667 | 1,533 | MasakhaNEWS, XL-Sum, UD Naija and more, from `pcm_results.zip`. `education` becomes `other`. 455 clear farming texts were relabelled agriculture |
| Yoruba | 5,966 | 1,500 | 1,500 | `yoruba_synthetic_clean.csv.gz`: Gemini-generated, labels unreviewed, no `other`. 1,034 training rows overlapped its own test sentences and were dropped |

`data/`, `inputs/`, `kaggle_*` and the model files contain corpus text. **Keep them private** and out of the public repository.

## Files

| Path | What it is |
|---|---|
| `build_domain_data.py` | Builds `data/domain_{train,dev,test}.jsonl` |
| `domain_model.py` | Labels, label aliases and the TF-IDF model class |
| `train_domain_classifier.py` | Trains the simple model; `--compare` also trains one model per language |
| `train_afroxlmr_domain.py` | Fine-tunes AfroXLMR. Options: `--lr`, `--epochs`, `--eval-every 0.5`, `--cap-per-cell`, `--seed`, and `--no-test` for dev-only sweeps |
| `ensemble_domain.py` | Averages AfroXLMR runs, mixes in the simple model and scores the test set |
| `predict_domain.py` / `predict_domain_afroxlmr.py` | Label any CSV or JSONL file |
| `build_client_notebook.py` | Rebuilds the client notebook from saved results |
| `kaggle/` | The Kaggle overnight job (`overnight.py`) and its metadata |
| `results/` | Simple-model results, plus `results/overnight/` for the tuned AfroXLMR runs and the final ensemble |
| `client_notebook/` | The presentation notebook and its result files (kept private, not in this repository) |

## Run it

```
pip install scikit-learn pandas openpyxl joblib scipy pyarrow
python build_domain_data.py
python train_domain_classifier.py --compare
python predict_domain.py in.csv out.csv --text-col Text --lang swa
```

**AfroXLMR on a GPU:**

```
python train_afroxlmr_domain.py --data-dir data --out-dir runs/seed13 --lr 2e-5 --epochs 4 --eval-every 0.5 --warmup 0.1 --cap-per-cell 15000 --seed 13
python ensemble_domain.py --runs runs/seed13 runs/seed14 runs/seed15 --tfidf model/domain_classifier.joblib
python predict_domain_afroxlmr.py in.csv out.csv --model-dir runs/seed14/model --text-col text --lang-col language
```

Both prediction scripts add these columns:
- `domains`, for example `finance|telecoms` or `other`;
- `primary_domain`;
- one `p_<domain>` score per domain.

**Where the trained models are**
- **Tuned runs:** the Kaggle notebook `bmsofficial/afroxlmr-domain-overnight`, under Output, in `runs/seed13`, `seed14` and `seed15`. The best single run is seed 14.
- **Earlier runs:** Google Drive, in `MyDrive/afroxlmr_domain_classifier/` and `..._v2/`.

## Adding data

Pass `build_domain_data.py --extra file.csv`, with columns:
- `language`: `swa`, `kin`, `pcm` or `yor`;
- `text`;
- `labels`: one domain, two joined with `|`, or `other`;
- `split`: `train`, `dev` or `test`.

Put hand-checked rows in `dev` or `test`. Then rebuild the data and retrain.

## What would improve it most

1. **Off-topic Kinyarwanda and Yoruba text** (sport, entertainment, politics) for training `other`.
2. **About 100 hand-labelled Yoruba texts** for a real Yoruba score.
3. **Hand-labelled Pidgin texts for law, telecoms and agriculture,** which are untested now.
4. **Hand-labelled two-domain examples.**
5. **A fresh, untouched test set** of about 100 texts per language for the final numbers.

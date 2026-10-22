"""Build the client presentation notebook (client_notebook/Domain_Classifier_Results.ipynb) from saved results.

Computes the final model's test predictions (3 AfroXLMR runs averaged, mixed 50/50 with TF-IDF, thresholds tuned
on dev), writes small CSVs next to the notebook, then executes the notebook so it opens with every output filled in.
    python build_client_notebook.py
"""
import collections, csv, json, os

import joblib
import nbformat as nbf
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor

from domain_model import DOMAINS, LABELS
from train_afroxlmr_domain import decide, eval_split, load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "client_notebook")
R = os.path.join(OUT, "results")
ON = os.path.join(HERE, "results", "overnight")


def main():
    os.makedirs(R, exist_ok=True)
    dev, test = load(os.path.join(HERE, "data"), "dev"), load(os.path.join(HERE, "data"), "test")
    train = load(os.path.join(HERE, "data"), "train")
    runs = [os.path.join(ON, "runs", f"seed{s}") for s in (13, 14, 15)]
    Px = np.mean([np.load(os.path.join(r, "test_probs.npy")) for r in runs], axis=0)
    cfg = json.load(open(os.path.join(ON, "ensemble_with_tfidf", "thresholds.json")))
    tf = joblib.load(os.path.join(HERE, "model", "domain_classifier.joblib"))
    Pt = tf.proba([r["text"] for r in test], [r["language"] for r in test])
    P = (1 - cfg["tfidf_weight"]) * Px + cfg["tfidf_weight"] * Pt
    preds, res = eval_split(P, test, cfg["thresholds"])

    with open(os.path.join(R, "final_test_predictions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "source", "human_labels", "predicted_labels", "top_label_correct"] + [f"p_{d}" for d in DOMAINS] + ["text"])
        for r, p, row in zip(test, preds, P):
            w.writerow([r["language"], r["source"], "|".join(r["labels"]), "|".join(p), p[0] in r["labels"]] + [round(float(x), 3) for x in row] + [r["text"]])
    with open(os.path.join(R, "final_per_label.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["language", "label", "precision", "recall", "f1", "test_texts"])
        for l, (per, s, _) in res.items():
            for o in per:
                w.writerow([l, o["label"], o["precision"], o["recall"], o["f1"], o["support"]])
    with open(os.path.join(R, "final_confusion_all_human.csv"), "w", encoding="utf-8", newline="") as f:
        conf = res["ALL"][2]; w = csv.writer(f); w.writerow(["human_label"] + LABELS)
        for g in LABELS:
            w.writerow([g] + [conf[(g, p)] for p in LABELS])

    def summ(path):   # language -> top-label accuracy from a report table
        out = {}
        for line in open(path, encoding="utf-8"):
            c = [x.strip() for x in line.split("|")]
            if len(c) > 5 and c[1] == "one model":   # TF-IDF report: | setup | language | n | top label | ...
                c = c[1:]
            if len(c) > 4 and c[1] in ("kin", "pcm", "swa", "yor", "ALL") and c[3].endswith("%"):
                out.setdefault(c[1], float(c[3].rstrip("%")))
        return out
    simple = summ(os.path.join(HERE, "results", "report.md"))
    final = {l: round(100 * s["primary_acc"], 1) for l, (_, s, _) in res.items()}
    final_f1 = {l: s["macro_f1"] for l, (_, s, _) in res.items()}
    before = {"swa": 77.1, "kin": 70.6, "pcm": 86.3, "yor": 92.7, "ALL": 82.5}   # AfroXLMR v2 (22 Oct, untuned, Colab)
    seeds = {l: [] for l in final}
    for r in runs:
        for l, v in summ(os.path.join(r, "report.md")).items():
            seeds[l].append(v)
    with open(os.path.join(R, "model_comparison.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["language", "simple_tfidf", "afroxlmr_untuned", "final_model", "final_macro_f1", "single_run_min", "single_run_max", "test_texts", "test_labels"])
        for l in ("swa", "kin", "pcm", "yor", "ALL"):
            n = sum(1 for r in test if (l == "ALL" and r["label_source"].startswith("gold")) or r["language"] == l)
            w.writerow([l, simple[l], before[l], final[l], final_f1[l], min(seeds[l]), max(seeds[l]), n, "synthetic (Gemini)" if l == "yor" else "human"])
    cnt = collections.Counter((r["language"], s) for s, rows in (("train", train), ("dev", dev), ("test", test)) for r in rows)
    with open(os.path.join(R, "data_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["language", "train_texts", "dev_texts", "test_texts", "test_labels_by"])
        for l in ("swa", "kin", "pcm", "yor"):
            w.writerow([l, cnt[(l, "train")], cnt[(l, "dev")], cnt[(l, "test")], "Gemini (synthetic, unreviewed)" if l == "yor" else "people"])
    sweep = json.load(open(os.path.join(ON, "sweep", "sweep_summary.json")))
    json.dump({"tfidf_weight": cfg["tfidf_weight"], "sweep": sweep}, open(os.path.join(R, "training_choices.json"), "w"), indent=1)

    nb = build()
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": OUT}})
    nbf.write(nb, os.path.join(OUT, "Domain_Classifier_Results.ipynb"))
    print("wrote", os.path.join(OUT, "Domain_Classifier_Results.ipynb"))


def build():
    choices = json.load(open(os.path.join(R, "training_choices.json")))
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    c = []
    c.append(md("""# One domain classifier for Swahili, Kinyarwanda, Nigerian Pidgin and Yoruba

**CMU-Africa MSIT practicum with Intron · Michael Sangwa · 22 October 2026**

One model reads a text in any of the four languages and labels it with **one or two** of
**agriculture, finance, health, law, telecoms**, or **other** when none of them fits.

**In short:**
- **82.7%** of human-labelled test texts get the right top label: **77.5%** for Swahili, **71.9%** for Kinyarwanda and **86.2%** for Pidgin.
- For Yoruba it is **96.3%**, but that test set was written by an AI, so expect lower on real Yoruba.
- The final model combines three fine-tuned **AfroXLMR** runs with a simple keyword-style model; the combination scored best on held-out tuning data.
- The remaining errors are mostly about **other**: off-topic texts given a domain, and domain texts missed as other. Fixing them needs more data, mainly off-topic Kinyarwanda and Yoruba text and more labelled Pidgin, rather than more modelling.

This notebook only reads saved results; it does not need the model or a GPU."""))
    c.append(code("""import json
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 140)
LANG = {"swa": "Swahili", "kin": "Kinyarwanda", "pcm": "Nigerian Pidgin", "yor": "Yoruba", "ALL": "All human-labelled"}
comparison = pd.read_csv("results/model_comparison.csv")
per_label = pd.read_csv("results/final_per_label.csv")
preds = pd.read_csv("results/final_test_predictions.csv")
data = pd.read_csv("results/data_summary.csv")
choices = json.load(open("results/training_choices.json"))"""))
    c.append(md("""## 1. The task

| Label | Covers |
|---|---|
| agriculture | crops, livestock, farming practice, food production |
| finance | banking, money, prices, mobile money, insurance, business |
| health | disease, treatment, nutrition, health services |
| law | courts, legislation, rights, legal procedure |
| telecoms | phones, networks, internet, broadcasting |
| other | anything outside the five domains (sport, entertainment, politics, ...) |

- **Multi-domain texts:** a text can get two domains when it clearly covers both, for example a mobile-money tax, which is finance and telecoms.
- **The same model** serves every language.
- **Igbo** was left out by team decision."""))
    c.append(md("## 2. The data\n\nTest texts were never used for training. Training rows that overlapped a test text were removed."))
    c.append(code("""data.assign(language=data.language.map(LANG)).rename(columns=lambda s: s.replace("_", " "))"""))
    c.append(md("""**Where it comes from**
- **Swahili:** licence-checked web, Wikipedia, legal and agricultural documents, and news. The test set is 240 passages labelled by hand.
- **Kinyarwanda:** the team's multidomain corpus. The test set is the team's hand-labelled gold sample.
- **Nigerian Pidgin:** news and speech transcripts. The test labels come from the datasets' own human topic labels; there are no law, telecoms or agriculture test texts.
- **Yoruba:** 10,000 sentences generated by Gemini, with unreviewed labels and no "other" examples. Its score is therefore reported separately."""))
    c.append(md(f"""## 3. How the model was built

1. **Simple baseline:** word and character patterns (TF-IDF) with one logistic-regression classifier per domain. It runs on a laptop.
2. **AfroXLMR:** `Davlan/afro-xlmr-base`, a multilingual model pre-trained on African languages, fine-tuned on a Kaggle T4 GPU.
3. **Tuning,** judged on held-out dev data only:
   - learning rate: {", ".join(f"{k} → {v:.3f}" for k, v in choices["sweep"]["sweep_dev_mean_macro_f1"].items())} (dev score), so **{choices["sweep"]["best_lr"]}** was chosen;
   - training data: about 95,000 texts;
   - training length: up to 4 passes over the data, keeping the best checkpoint;
   - repeats: 3 runs with different random seeds.
4. **Final model:** the three AfroXLMR runs averaged, then mixed **{int(choices["tfidf_weight"] * 100)}/{int(100 - choices["tfidf_weight"] * 100)}** with the simple model. The mix was chosen because it scored best on dev.
5. **Labelling rule:** a domain is assigned when its score passes a threshold tuned on dev for each language. If no domain passes, the text is "other"."""))
    c.append(md("""## How we measure

Every score is computed on test texts the model never saw in training. The test labels come from people for Swahili, Kinyarwanda and Pidgin, and from Gemini for Yoruba.

**Main scores**

| Metric | What it means | Example |
|---|---|---|
| **Top label correct** (headline) | Share of texts where the model's first label is one of the human labels | Human says "finance\\|telecoms", model says "telecoms": correct |
| **Exact label set** | Share where the model's full set of labels equals the human set; stricter | The same example counts as wrong |
| **Precision** (per label) | When the model says *law*, how often it really is law | |
| **Recall** (per label) | Of all real *law* texts, how many the model found | |
| **F1** (per label) | One score balancing precision and recall; 1.0 is perfect | |
| **Macro F1** | Average F1 over the labels that have test texts in that language. Each label counts equally, so a weak rare label (for example telecoms) pulls it down | This is why macro F1 (0.77) is lower than top-label accuracy (82.7%) |

**Used for decisions, on held-out dev data only (never the test set)**
- **Dev mean macro F1:** macro F1 on dev, averaged so each language counts equally. It chose the learning rate, the best checkpoint in each run, and the 50/50 mix with the simple model.
- **Per-domain F1 on dev:** sets each domain's threshold, the score a text needs to receive that label, separately for each language where there was enough dev data.

**Diagnostics**
- **Error table:** human label against the model's top label (section 4).
- **Two-domain counts:** section 6.
- **Spread across the three training runs:** about 2 points.

**Margin of error:** with test sets this size, a score is roughly ±5 points for Swahili (240 texts), ±4.5 for Kinyarwanda (385), ±2 for Pidgin (1,533) and ±1.6 for all human-labelled texts together (2,158), at about 95% confidence. Smaller differences between models can be chance.

**Not used:** weighted F1, which hides weak rare labels, and AUC or calibration scores, which a labelling tool doesn't need."""))
    c.append(md("## 4. Results\n\n**Top label correct:** the model's first label matches a human label."))
    c.append(code("""langs = ["swa", "kin", "pcm", "yor", "ALL"]
t = comparison.set_index("language").loc[langs]
fig, ax = plt.subplots(figsize=(10, 4.8))
series = [("simple_tfidf", "Simple model", "#2a78d6"), ("afroxlmr_untuned", "AfroXLMR, untuned", "#eb6834"), ("final_model", "Final model", "#1baf7a")]
width = 0.26
for i, (col, name, color) in enumerate(series):
    xs = [j + (i - 1) * width for j in range(len(langs))]
    bars = ax.bar(xs, t[col], width - 0.03, label=name, color=color)
    for b, v in zip(bars, t[col]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#333")
ax.set_xticks(range(len(langs)))
ax.set_xticklabels([LANG[l] + ("\\n(AI-written test)" if l == "yor" else "") for l in langs])
ax.set_ylim(0, 105); ax.set_ylabel("Top label correct (%)")
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
ax.set_title("Top label correct on test texts, by language", loc="left")
plt.tight_layout(); plt.show()"""))
    c.append(code("""t.assign(language=[LANG[l] for l in langs])[["language", "simple_tfidf", "afroxlmr_untuned", "final_model", "final_macro_f1", "single_run_min", "single_run_max", "test_texts", "test_labels"]] \\
 .rename(columns={"simple_tfidf": "simple %", "afroxlmr_untuned": "AfroXLMR untuned %", "final_model": "final %", "final_macro_f1": "final macro F1",
                  "single_run_min": "single run min %", "single_run_max": "single run max %", "test_texts": "test texts", "test_labels": "test labels"})"""))
    c.append(md("""**How to read this**
- **Margin of error:** the test sets are small (240 Swahili, 385 Kinyarwanda texts), so differences of 3–5 points can be chance.
- **Run-to-run luck:** the three AfroXLMR runs differ from one another by about 2 points. That is why the final model averages them.
- **Where AfroXLMR helps:** it is strongest on Swahili, whose test set is the most varied real-world text. On keyword-heavy text, the simple model is as good."""))
    c.append(md("### Per-label quality (final model)\n\n**F1** combines precision (how often a predicted label is right) and recall (how many true cases are found); 1.0 is perfect. Blank means no test texts for that label."))
    c.append(code("""f1 = per_label[per_label.language.isin(["swa", "kin", "pcm", "yor"])].copy()
f1.loc[f1.test_texts == 0, "f1"] = None
f1.pivot(index="label", columns="language", values="f1").loc[["agriculture", "finance", "health", "law", "telecoms", "other"], ["swa", "kin", "pcm", "yor"]] \\
  .rename(columns=LANG).round(2)"""))
    c.append(md("### Where the errors go (all human-labelled test texts)\n\nRows are the human label, columns the model's top label. The diagonal is correct."))
    c.append(code("""pd.read_csv("results/final_confusion_all_human.csv").set_index("human_label")"""))
    c.append(md("""Most errors involve **other**, in two directions of similar size:
- **Other row:** off-topic texts given a domain, often health, law or finance. In Kinyarwanda this happens because there was no off-topic Kinyarwanda text to train on.
- **Other column:** domain texts the model calls "other", mostly Pidgin finance and health news. Their domain is clear to a reader, but the training labels came mainly from keywords."""))
    c.append(md("## 5. Example predictions\n\nThese are held-out test texts, with the human label next to the model's labels."))
    c.append(code("""def show(lang, ok, n):
    d = preds[(preds.language == lang) & (preds.top_label_correct == ok)]
    d = d.groupby("human_labels", group_keys=False).head(1).head(n)
    return d.assign(text=d.text.str.slice(0, 140) + "…")[["human_labels", "predicted_labels", "text"]]

for lang in ["swa", "kin", "pcm", "yor"]:
    print(f"\\n{LANG[lang]}: correct")
    display(show(lang, True, 4))
    print(f"{LANG[lang]}: mistakes")
    display(show(lang, False, 2))"""))
    c.append(md("## 6. Two-domain texts"))
    c.append(code("""multi = preds[preds.human_labels.str.contains("\\\\|", regex=True)]
print("Human two-domain test texts:", len(multi), "| top label is one of the human labels:", int(multi.top_label_correct.sum()),
      "| both labels found:", int((multi.human_labels.str.split("|").apply(set) == multi.predicted_labels.str.split("|").apply(set)).sum()))
print("Texts the model gives two domains:", int(preds.predicted_labels.str.contains("\\\\|", regex=True).sum()), "of", len(preds))"""))
    c.append(md("""Two-domain output works, but the model rarely finds the second domain, because every training text carries a single label. More hand-labelled two-domain examples would fix this."""))
    c.append(md("""## 7. Limitations and next steps

| Gap | Effect | What would fix it |
|---|---|---|
| No off-topic Kinyarwanda or Yoruba training text | Off-topic texts are forced into a domain (Kinyarwanda "other" recall is low) | A few thousand off-topic texts per language (sport, entertainment, politics) |
| Yoruba labels and test set are AI-generated | The real Yoruba accuracy is unknown | About 100 hand-labelled Yoruba texts for testing |
| Pidgin test has no law, telecoms or agriculture | Those labels are untested for Pidgin | Hand-labelled examples for those domains |
| Training labels have one domain each | Second domains are rarely predicted | Hand-labelled two-domain examples |
| Small test sets | Scores carry about ±3–5 points of uncertainty | A fresh, untouched test set of about 100 texts per language for the final numbers |

**Planned use:** label the domain text that feeds Swahili/multilingual QA-pair generation, then keep refining the classifier as new data arrives.

## 8. Using the classifier

- **Code:** the team repository, `modeling/domain_classifier/`. Corpus text and model weights are kept private.
- **Trained models:** stored privately on Kaggle and Google Drive.
- **Labelling a file:** `predict_domain_afroxlmr.py in.csv out.csv --model-dir <model> --text-col text --lang-col language` adds `domains`, `primary_domain` and per-domain scores to each row."""))
    nb = nbf.v4.new_notebook(); nb.cells = c
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    main()

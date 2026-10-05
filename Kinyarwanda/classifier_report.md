
# Kinyarwanda Multidomain Classifier Report

## 1. Experiment Overview

A TF-IDF + Logistic Regression classifier was trained to classify
Kinyarwanda text into five domains:

- Agriculture
- Finance
- Health
- Law
- Telecoms

The manually annotated Gold Test set was excluded from both training
and internal validation to prevent test-set leakage.

The Gold Test also contains an `others` category. Because `others`
was not available as a training class, those examples are analyzed
separately rather than included in the five-class accuracy score.

---

## 2. Overall Performance

| Evaluation                  |   Samples | Accuracy   |   Macro F1 |
|:----------------------------|----------:|:-----------|-----------:|
| Internal validation         |     10027 | 94.39%     |     0.9451 |
| Human Gold Test (5 domains) |       476 | 86.97%     |     0.804  |

### Interpretation

The classifier achieved **94.39% accuracy** and a
**Macro F1-score of 0.9451** on the internal validation
set. This indicates strong performance in reproducing the domain
labels contained in the collected corpus.

On the independently human-annotated Gold Test, the classifier
achieved **86.97% accuracy** and a **Macro F1-score of
0.8040** for the five target domains.

The Gold Test accuracy is **7.41 percentage points
lower** than the internal validation accuracy. This difference is
important because the validation set reflects the original corpus
labels, whereas the Gold Test reflects independent human judgments.

The drop therefore suggests that reproducing the original corpus
labels is easier than reproducing human domain judgments. It may
also indicate ambiguity in some source labels and/or that the model
has learned source-specific writing patterns in addition to domain
semantics.

---

## 3. Internal Validation Performance by Domain

| Domain      |   Precision |   Recall |     F1 |   Support |
|:------------|------------:|---------:|-------:|----------:|
| health      |      0.9123 |   0.9209 | 0.9165 |      2123 |
| finance     |      0.9135 |   0.8915 | 0.9023 |      2156 |
| agriculture |      0.9278 |   0.9421 | 0.9349 |      2279 |
| telecoms    |      0.96   |   1      | 0.9796 |        48 |
| law         |      0.993  |   0.9915 | 0.9922 |      3421 |

The validation results are strong across all five domains.

**Law** obtained the highest validation F1-score
(0.9922).

**Telecoms** also obtained a very high validation F1-score
(0.9796), but this value should
be interpreted carefully because the validation set contains only
48 Telecoms examples,
substantially fewer than the other domains.

---

## 4. Human Gold Test Performance by Domain

| Domain      |   Precision |   Recall |     F1 |   Support |
|:------------|------------:|---------:|-------:|----------:|
| health      |      0.8488 |   0.7526 | 0.7978 |        97 |
| finance     |      0.7976 |   0.8171 | 0.8072 |        82 |
| agriculture |      0.8158 |   0.9118 | 0.8611 |       102 |
| telecoms    |      1      |   0.4211 | 0.5926 |        19 |
| law         |      0.9402 |   0.983  | 0.9611 |       176 |

The Human Gold Test provides a more demanding evaluation because its
labels were assigned manually and independently from the original
dataset labels.

**Law** remained the strongest Gold Test class, with an F1-score of
0.9611.

Agriculture, Finance, and Health achieved F1-scores of
0.8611,
0.8072, and
0.7978, respectively.

**Telecoms was the most difficult Gold Test class.** Its precision was
1.0000, while recall was only
0.4211, producing an F1-score of
0.5926.

This means that when the classifier predicted Telecoms it was usually
correct, but it failed to identify a substantial proportion of the
texts that the human annotator considered Telecoms. The small Gold
Test support of 19 examples
should also be considered when interpreting this result.

---

## 5. Analysis of the `others` Category

The Gold Test contains **100** examples manually
identified as `others`.

Because the classifier was trained only on the five target domains,
it was forced to assign each of these texts to one of those domains.

| Predicted_domain   |   Count | Percentage   |
|:-------------------|--------:|:-------------|
| health             |      41 | 41.0%        |
| finance            |      32 | 32.0%        |
| law                |      16 | 16.0%        |
| agriculture        |      10 | 10.0%        |
| telecoms           |       1 | 1.0%         |

The average classifier confidence was:

| Group | Average confidence |
|---|---:|
| Five target domains | 0.9152 |
| Human-labelled `others` | 0.7932 |

Confidence was lower for `others` than for examples belonging to the
five target domains. This suggests that confidence may contain useful
information for detecting ambiguous or out-of-domain texts.

However, the average confidence on `others`
(0.7932) is still relatively high. Therefore, a simple
confidence threshold should not automatically be treated as a reliable
`others` detector without further validation.

---

## 6. Key Findings

| Finding | Interpretation |
|---|---|
| Validation accuracy | 94.39% — strong reproduction of the original corpus labels |
| Validation Macro F1 | 0.9451 — strong performance across the five domains |
| Human Gold accuracy | 86.97% — lower than internal validation, showing that human evaluation is more challenging |
| Human Gold Macro F1 | 0.8040 — domain performance is less uniform on manually reviewed data |
| Accuracy gap | 7.41 percentage points between validation and Gold Test |
| Strongest Gold domain | Law |
| Most difficult Gold domain | Telecoms |
| Human-labelled `others` | 100 examples |
| Confidence on 5 domains | 0.9152 |
| Confidence on `others` | 0.7932 |

---

## 7. Conclusion

The classifier provides a strong baseline for automatic domain
classification of the collected Kinyarwanda corpus.

Its high internal validation performance shows that TF-IDF features
combined with Logistic Regression can effectively learn the five
domain labels in the collected data.

The lower performance against independently reviewed human labels,
however, demonstrates why manual validation is important. The
difference between corpus-label performance and Gold Test performance
suggests that some texts are semantically ambiguous or may not align
perfectly with the domain assigned by their original source.

The `others` analysis further shows that forcing every text into one
of five domains can produce confident predictions even for texts that
a human reviewer considers outside the target domains. Future work
could therefore investigate explicit abstention, confidence
calibration, or a separately trained `others` class using additional
independently labelled data.

# Domain classifier: retrained on seed v1, tested on Michael's human-reviewed passages

Model spec (both runs): word 1-2 gram TF-IDF + class-balanced logistic regression (last week's recipe, re-fitted because the saved pickle does not load under the installed scikit-learn).

**Test set:** all 100 passages Michael reviewed; his domain judgement gives the human-checked label. 97 had `domain_ok = YES` (the candidate label was confirmed), 17,118 words. Three had `domain_ok = NO` (a Bible-study passage filed under law, a milk-machine caption under finance, a university article under law) and are excluded from scoring.

**Training sets:** seed v1 (674 passages, 50,138 words; agriculture comes from the TARI and bean guides and law from TanzLII acts and gazette text, not Wikipedia; rule-filtered, licence-clear, not human-verified; 1 overlapping passage removed) versus last week's weak-label bootstrap sample (7,960 passages, keyword labels).

| Test subset | n | Trained on seed v1 | Trained on weak labels |
|---|---:|---:|---:|
| All human-reviewed, domain confirmed | 97 | **66.0%** | 100% |
| Gold (passages Michael kept) | 19 | 84.2% | 100% |
| Caption-like (rule flag) | 48 | 75.0% | 100% |
| Not caption-like | 49 | 57.1% | 100% |

Seed v1 model, per-class recall: agriculture 1.0, finance 0.58, health 0.56, law 0.25, telecoms 0.9 Main confusion: law passages predicted as finance (5), finance passages predicted as agriculture, law or telecoms (2, 2, 3). Replacing Wikipedia with real farming documents in the agriculture seed raised overall accuracy from 79.4% to 83.5%; adding TanzLII legal text as the law seed moved it to about 78-79% (now 79.4%), because the test's law passages (news and general text filed as law) look unlike statutes.

## How to read this
- The weak-label model's 100% is **not** evidence of quality. 13 of the 100 test passages have a near-duplicate (8-gram Jaccard at least 0.5) in its training data and 34 share some 8-gram overlap, and the test passages come from the same pool and the same keyword labelling. It reproduces keyword labels, as we saw earlier.
- The seed-trained model is the more honest number: trained on mostly Wikipedia and document text (1 near-duplicate of the test), tested on a different distribution. On non-caption text it gets 78% and on picture descriptions 79%. The drop from 92% on non-caption text came with the law seed switching to statute text, so the classifier now separates statutes well but not the news-style passages in the test.
- The gold subset is only 19 passages, so 94.7% (18 of 19) has a very wide uncertainty.
- No "others" class: both models force every passage into one of the five domains.

## Suggested use
Train on the seed (or a validated version of it), evaluate on passages from a *different* source than the training data, and report accuracy on non-caption text. Treat last week's 95% as agreement with keywords only.

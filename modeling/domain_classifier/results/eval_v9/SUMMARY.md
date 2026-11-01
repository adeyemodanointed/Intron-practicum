# v9 experiments: testing the advice (29 Oct 2026)

All numbers are top-label accuracy on the same 4,716 out-of-fold human-labelled texts as v8 ("validated" = the 1,042 team-validated texts). Full tables: `report.md`, `per_domain.csv`, `paired.csv`, `by_source.md`; TF-IDF-only comparison in `tfidf_ablation/`.

## What was tried

1. **Energy-based "other" gate** (the advice's biggest claim, +8 to +12 points on "other"). Tested on v8 probabilities with no retraining (`posthoc_other_gate.py`, `../eval_v8/other_gate.json`). Validated "other" went from 56.7% to at best 58.3%, and validated accuracy fell from 72.6% to 72.4%. The missed off-topic texts get confident domain scores (median 0.98), so an uncertainty gate cannot separate them.
2. **Label cleaning** (confident learning; `clean_labels.py`). Out-of-fold TF-IDF scores flag weak labels that another label clearly beats; 4,629 of 181,975 training rows (2.5%) were dropped, 4,288 of them Kinyarwanda (Afrivoice, KINNEWS). A spot check showed mostly real mistakes (a budget speech labelled agriculture, smartphone news labelled finance). Report: `../label_cleaning_v8/report.md`.
3. **FineWeb-2 off-topic text** (`build_domain_data.py --fineweb2`). 2,533 Pidgin and 3,000 Yoruba real web chunks, labelled `other`: religious sites always, other sites when they contain no domain word. Only human-written sites are used; most Yoruba FineWeb-2 sites are machine-translated and were left out under the no-AI-text rule.

## Results

| System | All texts | Validated | Validated balanced | Validated "other" |
|---|---:|---:|---:|---:|
| AfroXLMR-base v8 (delivered) | 83.9% | 72.6% | 75.4% | 56.7% |
| AfroXLMR-base v9 (FineWeb-2 + cleaning) | 84.1% | 72.6% | 74.9% | 61.0% |
| AfroXLMR + TF-IDF blend v9 | 83.9% | 72.6% | 74.8% | 62.7% |
| TF-IDF v8 | 83.0% | 69.3% | 71.1% | 64.7% |
| TF-IDF v8, cleaned | **83.9%** | 70.2% | 71.9% | 68.0% |
| TF-IDF v9 (FineWeb-2) | 83.4% | 69.1% | 69.4% | 71.3% |
| TF-IDF v9, cleaned | 83.6% | 69.1% | 70.0% | 70.0% |

- **Label cleaning helps TF-IDF:** +0.9 points on all texts (paired bootstrap p < 0.001), +2.4 on Kinyarwanda (p = 0.016), +1.0 on validated texts (p = 0.15, not significant).
- **FineWeb-2 off-topic text moves the boundary, it does not add accuracy.** "Other" goes up 4 to 7 points, but every domain loses a few points (Pidgin validated law: TF-IDF 51.8% to 40.7%), so the totals stay flat.
- **AfroXLMR v9 is not significantly different from v8** (+0.2 points on all texts, p = 0.66; +0.1 on validated texts, p = 0.94). It is better on validated "other" (56.7% to 61.0%), law (76.9% to 79.5%) and two-domain texts (both found 6 of 17, was 4), and worse on agriculture (83.3% to 77.8%) and telecoms (87.1% to 84.9%).
- The gains are not just from matching FineWeb-2: validated texts from FineWeb-2 rose 74.4% to 75.8%, the rest moved 72.1% to 71.8%.

**Recommendation:** keep AfroXLMR-base v8 as the delivered model and report v9 as an ablation ("label cleaning and more off-topic data: no significant change"). Both models sit at about 72-73% on validated texts whatever we change, which points to label-boundary noise in what counts as "other" versus a domain, not to the model.

## Next steps

1. **Error audit by native speakers** (`../../review/error_audit_validated_v8.csv`, 286 texts): mark whether each human label is right. Agree one rule with the team, e.g. "a domain only when the text is mainly about it; religious and political texts are `other` even with domain words". Fix the labels, re-score; this is now the most likely source of real gains.
2. **AfroXLMR on v8 cleaned only** (no FineWeb-2), 1.5 GPU hours once the Kaggle quota resets: cleaning helped TF-IDF without the domain trade-off. Kit: change `D` in `kaggle_v3/k_v9/v9.py` to point at a v8-cleaned dataset.
3. Serengeti (base size; licence not stated, research comparison only) and asymmetric loss, when GPU time allows.

The Colab AfroXLMR-large v8 run disconnected during stage 1 (about 56 minutes in) and saved nothing; it was not restarted, since large matched base earlier.

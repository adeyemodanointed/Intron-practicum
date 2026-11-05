# Methodology References

This reading list supports the methodology used in the multilingual domain
classification and punctuation-and-capitalisation (PnC) work. It is a guide for
the project report and presentation, not a claim that any paper exactly
duplicates this project.

## Short methodology statement

We formulate domain identification as supervised multi-class text
classification over five operational domains. Because labelled data are scarce
for the target languages, we use documented synthetic or weakly labelled
training data as a starting point, compare a lexical TF-IDF baseline with an
African-language pretrained transformer, fine-tune the transformer with LoRA,
and report performance on held-out manually reviewed external text separately
from development results. PnC is formulated as token-level punctuation and
capitalisation restoration from lower-cased, unpunctuated text.

## Core references for domain classification

| Methodological decision | Recommended source | What it supports |
|---|---|---|
| Use an African-language adapted encoder | Alabi et al. (2022), *Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning* | The AfroXLMR work directly motivates adapting XLM-R to African languages and evaluates it on downstream classification tasks. |
| Use multilingual pretrained representations rather than only sparse lexical features | Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning at Scale* | XLM-R demonstrates the value of multilingual pretraining and cross-lingual transfer, especially for low-resource languages. |
| Fine-tune efficiently with LoRA | Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models* | LoRA freezes the base parameters and trains small low-rank updates, reducing trainable parameters and memory cost. |
| Use synthetic text only as augmentation/seed data, not as proof of real-world performance | Chen et al. (2023), *An Empirical Survey of Data Augmentation for Limited Data Learning in NLP*; Evuru et al. (2024), *CoDa* | Augmentation is a recognized strategy under limited labelled data, but must be evaluated carefully. This supports our separate manual external evaluation. |
| Keep human-reviewed external data separate and guard against leakage | Elangovan et al. (2021), *Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation* | Similar or duplicated text across train and test can inflate reported performance. This motivates protected evaluation files, text de-duplication, and document-level splits. |
| Report macro-F1 alongside accuracy | Fouad and Weeds (2024), *SussexAI at ArAIEval Shared Task* | The paper explains why macro-F1 gives each class equal weight and is preferable when class sizes differ. |

## References for the PnC track

| Methodological decision | Recommended source | What it supports |
|---|---|---|
| Treat punctuation restoration as an ASR/post-processing problem | Alam, Khan, and Alam (2020), *Punctuation Restoration using Transformer Models for High- and Low-Resource Languages* | Establishes punctuation restoration as a transformer-based task relevant to low-resource languages and ASR text. |
| Fine-tune a bidirectional pretrained transformer for punctuation restoration | Courtland, Faulkner, and McElvain (2020), *Efficient Automatic Punctuation Restoration Using Bidirectional Transformers with Robust Inference* | Supports transformer fine-tuning as an efficient automatic punctuation-restoration approach. |
| Evaluate punctuation/capitalisation separately and include an ASR caveat | Znotins, Gruzitis, and Dargis (2025), *From Conversational Speech to Readable Text* | Supports low-resource transcript post-processing and highlights that clean-text restoration does not fully represent noisy ASR conditions. |

## Full citations and links

1. Jesujoba O. Alabi, David Ifeoluwa Adelani, Marius Mosbach, and Dietrich Klakow. 2022. [*Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning*](https://aclanthology.org/2022.coling-1.382/). COLING 2022.

2. Alexis Conneau, et al. 2020. [*Unsupervised Cross-lingual Representation Learning at Scale*](https://aclanthology.org/2020.acl-main.747/). ACL 2020.

3. Edward J. Hu, et al. 2022. [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685). ICLR 2022.

4. Jiaao Chen, Derek Tam, Colin Raffel, Mohit Bansal, and Diyi Yang. 2023. [*An Empirical Survey of Data Augmentation for Limited Data Learning in NLP*](https://aclanthology.org/2023.tacl-1.12/). *Transactions of the ACL*, 11:191–211.

5. Chandra Kiran Evuru, et al. 2024. [*CoDa: Constrained Generation based Data Augmentation for Low-Resource NLP*](https://aclanthology.org/2024.findings-naacl.238/). Findings of NAACL 2024.

6. Sangwon Elangovan, et al. 2021. [*Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation*](https://aclanthology.org/2021.eacl-main.113/). EACL 2021.

7. Tanvirul Alam, Akib Khan, and Firoj Alam. 2020. [*Punctuation Restoration using Transformer Models for High- and Low-Resource Languages*](https://aclanthology.org/2020.wnut-1.18/). W-NUT 2020.

8. Maury Courtland, Adam Faulkner, and Gayle McElvain. 2020. [*Efficient Automatic Punctuation Restoration Using Bidirectional Transformers with Robust Inference*](https://aclanthology.org/2020.iwslt-1.33/). IWSLT 2020.

9. Arturs Znotins, Normunds Gruzitis, and Roberts Dargis. 2025. [*From Conversational Speech to Readable Text: Post-Processing Noisy Transcripts in a Low-Resource Setting*](https://aclanthology.org/2025.wnut-1.15/). W-NUT 2025.

10. Mary Fouad and Julie Weeds. 2024. [*SussexAI at ArAIEval Shared Task: Mitigating Class Imbalance in Arabic Propaganda Detection*](https://aclanthology.org/2024.arabicnlp-1.55/). ArabicNLP 2024. Use this citation specifically for its explanation of macro-F1 under class imbalance; it is not an African-language domain-classification study.

## Wording to use in the report

> We compare a TF-IDF lexical baseline with AfroXLMR, an African-language
> adapted multilingual encoder. We apply low-rank adaptation (LoRA) to reduce
> the number of trainable parameters. Because labelled in-domain data are
> scarce, synthetic or weak labels are used only for training/development;
> model generalization is reported separately on manually reviewed external
> text. Macro-F1 is reported alongside accuracy to give equal importance to
> each domain.

## Important claims to avoid

- Do not claim that synthetic training data are equivalent to human labels.
- Do not describe the manually reviewed external sets as untouched final tests
  after using them to compare multiple model variants.
- Do not claim that clean-text PnC scores equal performance on noisy ASR
  transcripts; that requires a separate speech/ASR evaluation.

# Literature review: a multilingual domain classifier for African languages

*Prepared for the CMU-Africa MSIT practicum with Intron, 22 October 2026. Summaries come from each paper's abstract and the published record; check the full papers before quoting specific numbers.*

## 1. What we built, and what the literature needs to answer

The project needed **one model** that labels text in Swahili, Kinyarwanda, Nigerian Pidgin and Yoruba with one or two of **agriculture, finance, health, law, telecoms**, or **other**. The literature bears on six design questions:

1. **Benchmarks:** what exists for topic classification in these languages?
2. **Model:** which pretrained model to use?
3. **Baseline:** how strong are simple models?
4. **Labelling rule:** how to assign several labels, or "none of these"?
5. **Evaluation:** how to evaluate honestly, given seed variance, leakage and synthetic test data?
6. **Use:** how classifiers are used to filter and route web text.

## 2. Topic classification benchmarks for African languages

**MasakhaNEWS** (Adelani et al., 2023) is the main benchmark for news topic classification in African languages.
- **Coverage:** 16 languages widely spoken in Africa.
- **Baselines:** classical machine-learning models next to fine-tuned language models.
- **Motivation:** human-annotated topic classification datasets for African languages are scarce.
- **Use in our project:** its Pidgin and Swahili portions supply part of our Pidgin test labels (business → finance, health; other news categories → other).

**SIB-200** (Adelani et al., 2024) is a topic classification evaluation set in 200+ languages and dialects.
- **Construction:** the English Flores-200 sentences were annotated, and the labels transferred to their translations.
- **Finding:** a large gap between high- and low-resource languages. Languages unseen during pre-training, and those from under-represented families, perform worse.
- **Implication for us:** test every language separately rather than trusting one pooled score. Our report does this.

**KINNEWS and KIRNEWS** (Niyongabo et al., 2020) are news classification datasets for Kinyarwanda and Kirundi, with 14 categories.
- **Finding:** embeddings trained on Kinyarwanda, the higher-resourced language, transfer to Kirundi.
- **Possible use:** their non-target categories (sport, politics, entertainment, religion and others) are a ready source of **off-topic Kinyarwanda text**. That is exactly our largest gap. The licence should be checked first.

**The Swahili News Classification Dataset** (David, 2020, Zenodo) is a Swahili news corpus with six categories: national, international, finance (*uchumi*), health (*afya*), sport (*michezo*) and entertainment (*burudani*).
- **Use in our project:** finance and health news as training text, and sport/entertainment chunks as Swahili `other` text.

**Gap these benchmarks leave**
- **News taxonomies:** they use news categories such as politics, sport and business. None of them covers our exact domain set; **agriculture, law and telecoms are missing or merged into broader categories**.
- **"None of these" class:** none of them treats it as a target the way our use case requires.
- **Our label set:** this is why we built our own, mapping news categories onto it where that is defensible.

## 3. Pretrained models for African languages

**XLM-R** (Conneau et al., 2020) is a multilingual masked language model.
- **Training:** about 100 languages, on more than 2 TB of filtered CommonCrawl text.
- **Result:** large gains over multilingual BERT on cross-lingual benchmarks, with the biggest gains for low-resource languages such as Swahili.
- **Relation to our model:** it is the base model that AfroXLMR adapts.

**AfriBERTa** (Ogueji, Zhu and Lin, 2021) showed that a multilingual model trained **only** on 11 African languages, with under 1 GB of text, can match or beat mBERT and XLM-R on named-entity recognition and text classification. Small, Africa-focused data can be enough.

**AfroXLMR** (Alabi et al., 2022) applies *multilingual adaptive fine-tuning* to XLM-R.
- **Method:** continued pre-training on 17 high-resource African languages plus three other languages widely spoken in Africa.
- **Size:** removing tokens for non-African scripts halves the model size.
- **Results:** it is competitive with adapting to each language separately, at a fraction of the storage. It was evaluated on named-entity recognition, **news topic classification** and sentiment.
- **Our choice:** we used `Davlan/afro-xlmr-base` because it covers all four of our languages and was already used in the team's punctuation work.

**SERENGETI** (Adebara et al., 2023) covers 517 African languages and varieties.
- **Results:** state of the art on many of 20 datasets across eight tasks.
- **Relation to our project:** a candidate replacement for AfroXLMR, especially if more languages are added later.

## 4. Simple models are strong baselines

**fastText** (Joulin et al., 2017) showed that a linear model over word and n-gram features is often **on par with deep-learning classifiers** for text classification, while training in minutes on a CPU. That matches our finding:
- **Overall:** our TF-IDF and logistic-regression model scores within the margin of error of fine-tuned AfroXLMR (82.3% against 82.5–82.7%).
- **Kinyarwanda and synthetic Yoruba:** it beats AfroXLMR on these, which are the most keyword-heavy texts.

MasakhaNEWS also reports classical baselines next to fine-tuned models, for this reason.

**Implication:** combining the two types of model is reasonable, because they make different errors (keywords against meaning). Our final model mixes them 50/50, a weight chosen on dev data.

## 5. Several labels, and "none of these"

**Multi-label thresholds:** with one independent score per label, a decision threshold is needed for each label.
- **Fan and Lin (2007)** studied threshold-selection strategies such as SCut, which tunes one threshold per label on held-out data.
- **Later work:** a later study reports that per-class thresholds beat a single global threshold.
- **Our approach:** we tune one threshold per domain, and per language where dev has enough examples. A single shared threshold cost Swahili about 6 points in our first experiments.

**"Other" as open-set detection:** labelling a text "other" when no domain is confident is a form of out-of-distribution detection.
- **Hendrycks and Gimpel (2017)** established the simple baseline of thresholding the model's confidence. They also showed that confidence is often high even on inputs unlike the training data.
- **Our result matches:** where a language had **no off-topic training text** (Kinyarwanda, Yoruba), off-topic texts were confidently pushed into a domain.
- **Implication:** the literature suggests adding real out-of-domain training examples rather than relying on thresholds alone.

## 6. Evaluating honestly

**Seed variance:** Dodge et al. (2020) fine-tuned BERT hundreds of times, changing only the random seed.
- **Finding:** seeds alone cause substantial differences in results, through both weight initialisation and data order.
- **Early stopping:** stopping unpromising runs early improves results for a fixed compute budget.
- **What we did:** our three AfroXLMR runs differ by about 2 points overall. We report the spread, average the runs, and keep the best checkpoint by dev score (checked every half epoch).

**Train–test leakage:** Lee et al. (2022) found that near-duplicate text between training and evaluation sets is common in standard datasets, and inflates scores.
- **What we did:** we drop any training row that matches a dev or test text, or shares three or more word 8-grams with one.
- **Effect:** this removed 2,189 rows, including 1,034 from the Yoruba set, whose own splits overlapped.

**Synthetic labelled data:** Li et al. (2023) found that classifiers trained on LLM-generated data do worse as tasks become more subjective, and that synthetic data is less diverse than real data.
- **Our Yoruba data:** it is entirely Gemini-generated, including its test set, so its 96% score is optimistic.
- **Implication:** the literature supports reporting it separately and collecting human-labelled Yoruba texts.

## 7. Classifiers for filtering and routing text

Large web-data pipelines use lightweight classifiers to select text.
- **FineWeb-Edu** (Penedo et al., 2024) labelled 460,000 web pages with an LLM, trained a small classifier on those labels, and filtered trillions of tokens with it.
- **Known risks:** the classifier inherits the labelling model's biases. Other groups found that the English-trained classifier transferred poorly to Portuguese and trained their own.

Our use case is similar: route domain text into QA-pair generation. The same two lessons apply:
1. **Check classifier labels against human judgement in each language.** Our human test sets do this.
2. **Do not assume a classifier transfers to a language it was not trained on.** A Yoruba farming sentence came out "other" before Yoruba training data was added.

## 8. Summary: where this project fits

| Question | What the literature says | What we did |
|---|---|---|
| Benchmarks | News-topic sets exist (MasakhaNEWS, SIB-200, KINNEWS, Swahili News) but not for agriculture/law/telecoms or "other" | Built a six-label set; mapped news labels where defensible |
| Model | Africa-adapted XLM-R models beat generic ones on African tasks | AfroXLMR-base, fine-tuned once for all languages |
| Baseline | Linear n-gram models are often on par with deep models | TF-IDF baseline within the margin; ensembled with AfroXLMR |
| Several labels and "other" | Per-label thresholds; confidence-only out-of-domain detection is weak | Per-domain, per-language thresholds; real off-topic data identified as the main need |
| Evaluation | Seed variance, leakage and synthetic test data can mislead | Three seeds, n-gram leakage filter, separate synthetic Yoruba score, margins of error |

**Open directions suggested by the literature**
- Off-topic text from KINNEWS-style category datasets for "other".
- A SERENGETI comparison.
- Human-verified samples of the synthetic Yoruba data.

## References

- Adelani, D. I., et al. (2023). *MasakhaNEWS: News Topic Classification for African Languages.* AfricaNLP Workshop at ICLR 2023 and IJCNLP-AACL 2023. [arXiv:2304.09972](https://research.aston.ac.uk/files/103313920/2304.09972v1.pdf) · [Lancaster eprint](https://eprints.lancs.ac.uk/id/eprint/194287/)
- Adelani, D. I., et al. (2024). *SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects.* EACL 2024, pp. 226–245. [ACL Anthology](https://aclanthology.org/2024.eacl-long.14)
- Adebara, I., Elmadany, A., Abdul-Mageed, M., & Alcoba Inciarte, A. (2023). *SERENGETI: Massively Multilingual Language Models for Africa.* Findings of ACL 2023. [ACL Anthology](https://aclanthology.org/2023.findings-acl.97)
- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). *Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning.* COLING 2022, pp. 4336–4349. [ACL Anthology](https://aclanthology.org/2022.coling-1.382) · [model](https://huggingface.co/Davlan/afro-xlmr-base)
- Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020, pp. 8440–8451. [ACL Anthology](https://aclanthology.org/2020.acl-main.747)
- David, D. (2020). *Swahili: News Classification Dataset* (v0.2). Zenodo. [doi:10.5281/zenodo.5514203](https://zenodo.org/record/5514203)
- Dodge, J., et al. (2020). *Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping.* [arXiv:2002.06305](https://arxiv.org/abs/2002.06305v1)
- Fan, R.-E., & Lin, C.-J. (2007). *A Study on Threshold Selection for Multi-label Classification.* Technical report, National Taiwan University. Cited via [utiml documentation](https://rdrr.io/github/rivolli/utiml/man/scut_threshold.html)
- Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR 2017. [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)
- Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2017). *Bag of Tricks for Efficient Text Classification.* EACL 2017, pp. 427–431. [ACL Anthology](https://preview.aclanthology.org/setup/E17-2068)
- Lee, K., et al. (2022). *Deduplicating Training Data Makes Language Models Better.* ACL 2022. [arXiv:2107.06499](https://arxiv.org/pdf/2107.06499)
- Li, Z., Zhu, H., Lu, Z., & Yin, M. (2023). *Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations.* EMNLP 2023, pp. 10443–10461. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.647)
- Niyongabo, R. A., Qu, H., Kreutzer, J., & Huang, L. (2020). *KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi.* COLING 2020, pp. 5507–5521. [ACL Anthology](https://aclanthology.org/2020.coling-main.480)
- Ogueji, K., Zhu, Y., & Lin, J. (2021). *Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages.* MRL Workshop at EMNLP 2021. [PDF](https://cs.uwaterloo.ca/~jimmylin/publications/Ogueji_etal_MRL2021.pdf)
- Penedo, G., et al. (2024). *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* [arXiv:2406.17557](https://arxiv.org/pdf/2406.17557)

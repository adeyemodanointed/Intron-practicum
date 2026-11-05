# Yoruba source inventory

This file records candidate sources considered for Yoruba domain-classification
training.  A source is not automatically suitable merely because it contains
Yoruba text: its topic labels, licence, intended use, and overlap with protected
evaluation data must also be checked.

## AfriDocMT

- **Dataset:** [`masakhane/AfriDocMT`](https://huggingface.co/datasets/masakhane/AfriDocMT)
- **Yoruba field:** `yo`
- **Available topical subsets:** `Health` and `Tech`, each offered as document-
  and sentence-level data with `train`, `dev`, and `test` splits.
- **Coverage:** 334 Health documents and 271 Tech documents.  The documents are
  human translations into Yoruba and four other African languages.
- **Use in this project:**
  - `sentence/Health` can be used as real, dataset-labelled **Health** training
    data.  Preserve the dataset split and source-document relationship.
  - `Tech` must **not** be automatically renamed Telecommunications.  It is a
    broader IT category; inspect or review it before selecting genuine telecom
    examples.
  - It does not provide Agriculture, Finance, or Law labels.
- **Licence:** Health is CC BY-NC-SA 3.0; Tech is CC BY-NC 4.0.  Attribute the
  dataset and follow the non-commercial/share-alike terms.
- **Suggested metadata:** `Source=AfriDocMT`, `Domain=Health`,
  `Domain_Method=dataset_label`, `Label_Quality=source_gold`.

## FLORES+

- **Dataset:** [`openlanguagedata/flores_plus`](https://huggingface.co/datasets/openlanguagedata/flores_plus)
- **Yoruba configuration:** `yor_Latn`
- **Content:** 997 `dev` and 1,012 `devtest` Yoruba sentences, aligned with
  translations in other languages.  Source material comes from Wikinews,
  Wikijunior, and Wikivoyage and includes broad fields such as `domain` and
  `topic`.
- **Do not use for training:** FLORES+ is an evaluation benchmark, and its data
  card explicitly says it should not be used as training data.
- **Licence:** CC BY-SA 4.0; access requires accepting the dataset terms.
- **Potential future use:** only as a separately documented general-language or
  translation evaluation resource, if this does not overlap with the project’s
  own evaluation protocol.

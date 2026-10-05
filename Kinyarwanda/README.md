# Kinyarwanda Multidomain Data Pipeline

This repository contains the notebooks used to collect, clean, validate, and classify Kinyarwanda text across five target domains:

- Health
- Agriculture
- Finance
- Telecommunications
- Law

The work is organized into three main notebooks.

## Notebooks

### 1. `data-mining (1).ipynb`

Handles the collection and preparation of the multidomain Kinyarwanda corpus.

Main tasks include:

- Loading data from the selected sources
- Assigning source-based domain labels
- Removing duplicate records
- Checking and repairing encoding issues
- Counting tokens
- Building candidate datasets for the five domains
- Creating samples for manual review
- Exporting the final multidomain corpus

The main output is the cleaned Kinyarwanda multidomain dataset used in the later stages.

---

### 2. `capitalization-and-punctuation.ipynb`

Performs text-quality analysis focused on capitalization, punctuation, spacing, and other formatting issues in the collected Kinyarwanda text.

Main tasks include:

- Checking whether sentences begin with appropriate capitalization
- Checking sentence-ending punctuation
- Detecting spacing problems
- Examining apostrophe usage
- Identifying potentially malformed text
- Supporting manual quality review before downstream use

This notebook helps ensure that the text is suitable for later speech and language processing tasks.

---

### 3. `classifier.ipynb`

Trains and evaluates the Kinyarwanda domain classifier.

The classifier predicts one of five domains:

- Health
- Agriculture
- Finance
- Telecommunications
- Law

The notebook:

- Removes the manually annotated Gold Test examples from training
- Splits the remaining corpus into training and validation sets
- Extracts word and character TF-IDF features
- Trains a Logistic Regression classifier
- Evaluates performance using Accuracy, Macro F1, per-class metrics, and confusion matrices
- Evaluates the classifier against the independently annotated Gold Test set
- Analyzes examples manually labelled as `others`
- Saves the trained classifier and evaluation outputs

The final classifier package is stored in:

`kinyarwanda_classifier_final.zip`

It contains the trained model, evaluation reports, predictions, metrics, confusion matrices, and supporting metadata.

---

## Pipeline

The notebooks follow this general workflow:

`Data Collection → Text Quality Analysis → Domain Classification`

Together, they provide a reproducible pipeline for preparing and evaluating a multidomain Kinyarwanda text corpus.
"""Swahili domain classifier: word 1-2 gram TF-IDF + class-balanced logistic regression.
Train: Data.csv rows with Split == train. Predict: Evaluation.csv rows (the dev split) -> writes Domain_Pred back into Evaluation.csv.
Dev rows are human-kept passages; their domain label is the reviewer-confirmed domain. Usage: python train_domain_classifier.py Data.csv Evaluation.csv
"""
import csv, sys, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
data, ev = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(data, encoding="utf-8-sig")) if r["Split"] == "train"]
X = [r["Text"] for r in rows]; y = [r["Domain"] for r in rows]
m = Pipeline([("tfidf", TfidfVectorizer(analyzer="word", lowercase=True, max_df=0.98, max_features=75000, min_df=2, ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b")),
              ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=300, random_state=42))]).fit(X, y)
print("trained on", len(X), "passages;", dict(collections.Counter(y)))
E = list(csv.DictReader(open(ev, encoding="utf-8-sig"))); pred = m.predict([r["Text"] for r in E])
for r, p in zip(E, pred): r["Domain_Pred"] = p
with open(ev, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(E[0].keys())); w.writeheader(); w.writerows(E)
print("wrote Domain_Pred for", len(E), "rows")

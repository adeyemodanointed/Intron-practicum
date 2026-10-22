"""One multilingual, multi-label domain classifier (kept in its own module so a saved model can be
loaded by predict_domain.py).

Labels: agriculture, finance, health, law, telecoms, or other.
A text gets every domain whose probability reaches that domain's threshold (at most MAX_LABELS,
highest first). If no domain reaches its threshold the text is `other`. Thresholds are tuned per language
where that language has enough dev examples of the domain; otherwise the shared ("all") threshold is used.

Features: word 1-2 grams + character 3-5 grams (TF-IDF) on lower-cased text with diacritics removed,
plus a language tag token so the shared model can learn per-language priors. Optional augment=True: every feature
also gets a per-language copy ("shared + language-specific" feature augmentation): the model learns weights shared
by all languages plus a per-language adjustment. A language unseen in training uses the shared weights only.
One binary logistic regression per domain (one-vs-rest), class-balanced, with sample weights.
"""
import re
import unicodedata

import numpy as np

DOMAINS = ["agriculture", "finance", "health", "law", "telecoms"]
OTHER = "other"
LABELS = DOMAINS + [OTHER]
MAX_LABELS = 2

ALIASES = {
    "agric": "agriculture", "agriculture": "agriculture",
    "finance": "finance", "fin": "finance",
    "health": "health",
    "law": "law", "legal": "law",
    "telecoms": "telecoms", "telecom": "telecoms", "telecommunications": "telecoms", "telecommunication": "telecoms",
    "other": OTHER, "others": OTHER, "none": OTHER,
    "education": OTHER,   # Pidgin/Igbo pipelines used education; it is not one of the five domains
}


def norm_labels(value):
    """'Finance, Telecommunications' / 'finance|law' / ['law'] -> sorted unique canonical labels."""
    parts = value if isinstance(value, (list, tuple)) else re.split(r"[|,;/]+", str(value or ""))
    out = []
    for p in parts:
        p = p.strip().lower()
        if not p:
            continue
        if p not in ALIASES:
            raise ValueError(f"unknown domain label: {p!r}")
        if ALIASES[p] not in out:
            out.append(ALIASES[p])
    domains = [d for d in out if d != OTHER]
    return domains or [OTHER]


def fold(text):
    t = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


class DomainClassifier:
    def __init__(self, C=4.0, word_max=200_000, char_max=300_000, augment=False):
        self.C, self.word_max, self.char_max, self.augment = C, word_max, char_max, augment
        self.langs = []
        self.thresholds = {"all": {d: 0.5 for d in DOMAINS}}   # plus optional per-language overrides
        self.models = {}

    def _prep(self, texts, langs):
        return [f"langtag{l} {fold(t)}" for t, l in zip(texts, langs)]

    def _x(self, texts, langs, fit=False):
        from scipy.sparse import hstack
        p = self._prep(texts, langs)
        if fit:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                                        max_features=self.word_max, token_pattern=r"(?u)\b\w[\w'-]*\b", dtype=np.float32)
            self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                        max_features=self.char_max, dtype=np.float32)
            self.langs = sorted(set(langs)) if self.augment else []
            X = hstack([self.word.fit_transform(p), self.char.fit_transform(p)]).tocsr()
        else:
            X = hstack([self.word.transform(p), self.char.transform(p)]).tocsr()
        if not self.langs:
            return X
        from scipy.sparse import diags
        lang = np.asarray(langs)
        return hstack([X] + [diags((lang == l).astype(np.float32)) @ X for l in self.langs]).tocsr()

    def fit(self, texts, langs, labels, weights):
        from sklearn.linear_model import LogisticRegression
        X = self._x(texts, langs, fit=True)
        w = np.asarray(weights, dtype=float)
        for d in DOMAINS:
            y = np.array([d in ls for ls in labels], dtype=int)
            self.models[d] = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000, solver="liblinear").fit(X, y, sample_weight=w)
        return self

    def proba(self, texts, langs):
        X = self._x(texts, langs)
        return np.column_stack([self.models[d].predict_proba(X)[:, 1] for d in DOMAINS])

    def th_for(self, lang):
        th = dict(self.thresholds["all"])
        th.update(self.thresholds.get(lang, {}))
        return th

    def decide(self, P, langs, max_labels=MAX_LABELS):
        out = []
        for row, lang in zip(P, langs):
            th = self.th_for(lang)
            order = np.argsort(-row)
            picked = [DOMAINS[k] for k in order if row[k] >= th[DOMAINS[k]]][:max_labels]
            out.append(picked or [OTHER])
        return out

    def predict(self, texts, langs):
        return self.decide(self.proba(texts, langs), langs)

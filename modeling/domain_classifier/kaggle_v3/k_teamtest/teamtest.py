"""Probabilities of the saved final AfroXLMR models (outputs of domain-v3-final-large / -base) and the TF-IDF final model on the independent team
test set (teamtest_domain_teamtest.jsonl), on CPU. Thresholds are applied locally (cv_collect.py --package)."""
import glob, json, os, sys

import numpy as np
import torch

SRC = os.path.dirname(glob.glob("/kaggle/input/**/train_afroxlmr_domain.py", recursive=True)[0])
sys.path.insert(0, SRC)
from train_afroxlmr_domain import predict_proba  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

rows = [json.loads(l) for l in open(f"{SRC}/teamtest_domain_teamtest.jsonl", encoding="utf-8")]
os.makedirs("/kaggle/working/teamtest", exist_ok=True)
torch.set_num_threads(os.cpu_count())
for cfg in sorted(glob.glob("/kaggle/input/**/final/model/config.json", recursive=True)):
    d = os.path.dirname(cfg)
    name = "large" if "final-large" in d or "/large/" in d else "base"
    tok, model = AutoTokenizer.from_pretrained(d), AutoModelForSequenceClassification.from_pretrained(d)
    ids = tok([r["text"] for r in rows], truncation=True, max_length=256)["input_ids"]
    P = predict_proba(model, ids, tok.pad_token_id, torch.device("cpu"), batch=32)
    np.save(f"/kaggle/working/teamtest/{name}_probs.npy", P)
    print(name, d, P.shape, flush=True)
for jl in glob.glob("/kaggle/input/**/tfidf6/final/domain_classifier.joblib", recursive=True):   # same sklearn as training
    import joblib
    from domain_model import DomainClassifier  # noqa: F401  (class needed to unpickle)
    m = joblib.load(jl)
    np.save("/kaggle/working/teamtest/tfidf6_probs.npy", m.proba([r["text"] for r in rows], [r["language"] for r in rows]))
    print("tfidf6", jl, flush=True)
json.dump([r["id"] for r in rows], open("/kaggle/working/teamtest/ids.json", "w"))

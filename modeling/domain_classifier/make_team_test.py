"""Independent test set from the team's Practicum sheet (Manual Validation + the Yoruba tab): texts a person labelled,
from other sources than our test sets (WURA, FineWeb-2, Wikipedia, XL-Sum, MAFAND, MasakhaNEWS, Gazette).

    python make_team_test.py      # -> data_teamtest/domain_teamtest.jsonl

This is the official final test set for all four languages. Texts that are also in our cross-validation pool (all 360
Swahili ones = our Swahili human set, plus a few others) are kept and flagged in_eval_pool; build_domain_data.py
--heldout removes them from the pool used to train the final model. Each row also records whether it overlaps (exact
text or 3+ shared word 8-grams) the training data of each data version: overlap_<version> = true/false, so earlier
models are scored only on texts they never trained on.
"""
import collections, json, os, re

from domain_model import norm_labels

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "inputs", "external", "team_sheet", "team_sheet_labels.jsonl")
VERSIONS = ["v3", "v3m", "v4", "v5", "v7"]


def grams(t, n=8):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


key = lambda t: re.sub(r"\W+", " ", t.lower()).strip()


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    rows = [r for r in rows if r["tab"] in ("manual_validation", "sheet11_yoruba") and r["label"].strip()]
    ev = [json.loads(l) for l in open(os.path.join(HERE, "data_v5", "domain_eval.jsonl"), encoding="utf-8")]
    ev_keys, ev_g = {key(e["text"]) for e in ev}, set().union(*(grams(e["text"]) for e in ev))
    train = {}
    for v in VERSIONS:
        tr = [json.loads(l)["text"] for l in open(os.path.join(HERE, f"data_{v}", "domain_train.jsonl"), encoding="utf-8")]
        train[v] = ({key(t) for t in tr}, set().union(*(grams(t) for t in tr)))
    out, skipped = [], collections.Counter()
    for r in rows:
        k, g = key(r["text"]), grams(r["text"])
        in_pool = k in ev_keys or len(g & ev_g) >= 3
        skipped[r["language"]] += in_pool
        rec = {"id": f"team_{r['tab']}_{r['row']}", "language": r["language"], "text": re.sub(r"\s+", " ", r["text"]).strip(),
               "labels": norm_labels(r["label"]), "split": "test", "source": r["source"], "label_source": "gold:team_manual_validation",
               "fold": -1, "in_eval_pool": in_pool}
        for v, (keys, gr) in train.items():
            rec[f"overlap_{v}"] = k in keys or len(g & gr) >= 3
        out.append(rec)
    os.makedirs(os.path.join(HERE, "data_teamtest"), exist_ok=True)
    with open(os.path.join(HERE, "data_teamtest", "domain_teamtest.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("also in our cross-validation pool (kept, flagged):", dict(skipped))
    print("kept:", dict(collections.Counter(r["language"] for r in out)))
    for v in VERSIONS:
        print(f"overlap with {v} training:", dict(collections.Counter(r["language"] for r in out if r[f"overlap_{v}"])))
    print("labels:", sorted(collections.Counter((r["language"], "|".join(r["labels"])) for r in out).items()))


if __name__ == "__main__":
    main()

"""Domain classification evaluation per language per domain.
Input CSV columns: Language, Domain (reference), Domain_Pred (model output). Rows with an empty Domain_Pred are skipped.
Prints and optionally writes precision, recall, F1 and support per domain, plus overall accuracy and macro F1, per language.
Usage: python evaluate_domain_class.py Evaluation.csv [out.csv]
"""
import csv, sys, collections
def main(path, out=None):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if (r.get("Domain_Pred") or "").strip()]
    langs = sorted({r["Language"] for r in rows}); table = []
    for lang in langs:
        sub = [r for r in rows if r["Language"] == lang]; doms = sorted({r["Domain"] for r in sub} | {r["Domain_Pred"] for r in sub})
        acc = sum(r["Domain"] == r["Domain_Pred"] for r in sub) / len(sub); f1s = []
        for d in doms:
            tp = sum(r["Domain"] == d and r["Domain_Pred"] == d for r in sub); fp = sum(r["Domain"] != d and r["Domain_Pred"] == d for r in sub); fn = sum(r["Domain"] == d and r["Domain_Pred"] != d for r in sub)
            p = tp / (tp + fp) if tp + fp else 0.0; r_ = tp / (tp + fn) if tp + fn else 0.0; f = 2 * p * r_ / (p + r_) if p + r_ else 0.0
            if tp + fn: f1s.append(f)
            table.append({"language": lang, "domain": d, "precision": round(p, 3), "recall": round(r_, 3), "f1": round(f, 3), "support": tp + fn})
        table.append({"language": lang, "domain": "ALL (accuracy / macro F1)", "precision": round(acc, 3), "recall": "", "f1": round(sum(f1s) / len(f1s), 3), "support": len(sub)})
    if out:
        with open(out, "w", encoding="utf-8", newline="") as f: w = csv.DictWriter(f, fieldnames=["language", "domain", "precision", "recall", "f1", "support"]); w.writeheader(); w.writerows(table)
    for t in table: print(t)
if __name__ == "__main__": main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

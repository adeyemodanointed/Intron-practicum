"""Punctuation and capitalisation evaluation per language per domain.
Input CSV columns: Language, Domain, Source, Text (reference), Text_No_Punct, Hypothesis (model output). Rows with an empty Hypothesis are skipped.
Words are whitespace words. Punctuation label of a word: COMMA, PERIOD, QUESTION, EXCL, OTHER (; : ellipsis), or O. Capitalisation label: LOWER, TITLE, UPPER, MIXED.
A row is scored only if the reference and hypothesis contain the same words after removing punctuation and ignoring case (otherwise it is counted as 'misaligned').
Usage: python evaluate_pnc.py Evaluation.csv [out.csv]
"""
import csv, re, sys, collections, unicodedata
LEAD = re.compile(r"^[(\[\"«]+"); TRAIL = re.compile(r"[,.;:?!…)\"\]»]+$")
PL = ["COMMA", "PERIOD", "QUESTION", "EXCL", "OTHER"]
def split_word(raw):
    pre = LEAD.match(raw); pre = pre.group(0) if pre else ""
    rest = raw[len(pre):]; post = TRAIL.search(rest); post = post.group(0) if post else ""
    return rest[: len(rest) - len(post)], post
def punct_label(post):
    for ch in post:
        if ch == ",": return "COMMA"
        if ch == "?": return "QUESTION"
        if ch == "!": return "EXCL"
        if ch in ";:…": return "OTHER"
        if ch == ".": return "OTHER" if "..." in post else "PERIOD"
    return "O"
def cap_label(w):
    L = [c for c in w if c.isalpha()]
    if not L: return "LOWER"
    if len(L) > 1 and all(c.isupper() for c in L): return "UPPER"
    if L[0].isupper(): return "TITLE" if not any(c.isupper() for c in L[1:]) else "MIXED"
    return "MIXED" if any(c.isupper() for c in L) else "LOWER"
def labels(text):
    text = unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip()); out = []
    for raw in text.split(" "):
        core, post = split_word(raw)
        if not core and out: out[-1] = (out[-1][0], out[-1][1] if out[-1][1] != "O" else punct_label(post), out[-1][2]); continue
        out.append((core, punct_label(post), cap_label(core)))
    return out
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
def main(path, out=None):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if not (r.get("Hypothesis") or "").strip(): continue
        key = (r["Language"], r["Domain"]); ref = labels(r["Text"]); hyp = labels(r["Hypothesis"])
        if [w.lower() for w, _, _ in ref] != [w.lower() for w, _, _ in hyp]: agg[key]["misaligned_rows"] += 1; agg[(r["Language"], "ALL")]["misaligned_rows"] += 1; continue
        for k in (key, (r["Language"], "ALL")):
            a = agg[k]; a["rows"] += 1
            for (w, rp, rc), (_, hp, hc) in zip(ref, hyp):
                a["words"] += 1; a["punct_correct"] += rp == hp; a["cap_correct"] += rc == hc
                for c in PL:
                    a[c + "_tp"] += rp == c and hp == c; a[c + "_fp"] += rp != c and hp == c; a[c + "_fn"] += rp == c and hp != c
    cols = ["language", "domain", "rows", "misaligned_rows", "words", "punct_accuracy", "cap_accuracy"] + [f"{c}_{m}" for c in PL for m in ("P", "R", "F1")] + ["macro_F1_punct"]
    table = []
    for (lang, dom), a in sorted(agg.items()):
        row = {"language": lang, "domain": dom, "rows": a["rows"], "misaligned_rows": a["misaligned_rows"], "words": a["words"],
               "punct_accuracy": round(a["punct_correct"] / a["words"], 4) if a["words"] else "", "cap_accuracy": round(a["cap_correct"] / a["words"], 4) if a["words"] else ""}
        f1s = []
        for c in PL:
            p, r_, f = prf(a[c + "_tp"], a[c + "_fp"], a[c + "_fn"]); row.update({f"{c}_P": round(p, 3), f"{c}_R": round(r_, 3), f"{c}_F1": round(f, 3)})
            if a[c + "_tp"] + a[c + "_fn"]: f1s.append(f)
        row["macro_F1_punct"] = round(sum(f1s) / len(f1s), 3) if f1s else ""; table.append(row)
    if out:
        with open(out, "w", encoding="utf-8", newline="") as f: w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(table)
    for t in table: print({k: t[k] for k in ("language", "domain", "rows", "misaligned_rows", "words", "punct_accuracy", "cap_accuracy", "macro_F1_punct")})
if __name__ == "__main__": main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

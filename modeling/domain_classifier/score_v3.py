"""Score every v3 system the same way once the Kaggle outputs are downloaded (kaggle_v3/orchestrate.py puts them in
results/kaggle_v3/<kernel>/). Systems whose files are missing are skipped, so this can be rerun as results arrive.

    python score_v3.py            # -> results/eval_v3/{predictions/<system>/, main/, ablation/, llm/, multi/}

  predictions/<system>/test_predictions.csv  out-of-fold predictions (cv_collect.py) or LLM answers
  ablation/   AfroXLMR-base step by step and TF-IDF, on the texts every system labelled (commercial-safe eval set)
  main/       final models vs TF-IDF vs every LLM on all human-labelled texts (incl. real Yoruba), paired tests vs TF-IDF
  llm/        the same, paired tests against the best LLM
"""
import csv, glob, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.join(HERE, "results", "kaggle_v3")
OUT = os.path.join(HERE, "results", "eval_v3")
EV3, EV3C = os.path.join(HERE, "data_v3", "domain_eval.jsonl"), os.path.join(HERE, "data_v3c", "domain_eval.jsonl")
EV4 = os.path.join(HERE, "data_v4", "domain_eval.jsonl")
EV5 = os.path.join(HERE, "data_v5", "domain_eval.jsonl")
PRED = os.path.join(OUT, "predictions")

# name: (eval file, run dirs averaged, optional mix dir)
CV_SYSTEMS = {
    "tfidf_v2_setup": (EV3C, [os.path.join(HERE, "results", "cv", "tfidf5_c", "stage1")], None),
    "tfidf_c_gold": (EV3C, [f"{K}/domain-v3-tfidf/cv/tfidf5_c"], None),
    "tfidf6_full": (EV3, [f"{K}/domain-v3-tfidf/cv/tfidf6"], None),
    "base_a_v2_setup": (EV3C, [f"{K}/domain-v3-ablations/runs/c_sigmoid5/stage1"], None),
    "base_b_plus_human_labels": (EV3C, [f"{K}/domain-v3-ablations/runs/c_sigmoid5/stage2"], None),
    "base_c_plus_other_output": (EV3C, [f"{K}/domain-v3-ablations/runs/c_multi6/stage2"], None),
    "base_d_plus_news_data": (EV3, [f"{K}/domain-v3-ablations/runs/v3_multi6/stage2"], None),
    "base_e_plus_multi_domain": (EV3, [f"{K}/domain-v3-final-base/runs/base/stage2"], None),
    "large_final": (EV3, [f"{K}/domain-v3-final-large/runs/large/stage2"], None),
    "large_plus_tfidf": (EV3, [f"{K}/domain-v3-final-large/runs/large/stage2"], f"{K}/domain-v3-tfidf/cv/tfidf6"),
    "base_v4_new_data": (EV4, [f"{K}/domain-v4-base/runs/v4_multi6/stage2"], None),
    "base_v4_new_data_mix": (EV4, [f"{K}/domain-v4-base/runs/v4_multi6_mix/stage2"], None),
    "base_v5_plus_afrivoice": (EV5, [f"{K}/domain-v5-base/runs/v5_multi6/stage2"], None),
}
LLM_DIRS = [d for d in sorted(glob.glob(f"{K}/domain-v3-llm-*/llm/*shot")) + sorted(glob.glob(os.path.join(HERE, "results", "llm", "*shot")))
            if not os.path.basename(d).startswith("multi_candidates")]   # those label training texts, not the eval set
ABLATION = ["tfidf_v2_setup", "tfidf_c_gold", "base_a_v2_setup", "base_b_plus_human_labels", "base_c_plus_other_output",
            "base_d_plus_news_data", "base_e_plus_multi_domain", "large_final"]
MAIN = ["tfidf6_full", "base_d_plus_news_data", "base_e_plus_multi_domain", "large_final", "large_plus_tfidf",
        "base_v4_new_data", "base_v4_new_data_mix", "base_v5_plus_afrivoice"]


def run(*args):
    print(">", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=HERE, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def have(runs, mix):
    ok = lambda d: os.path.exists(os.path.join(d, "eval_probs.npy")) or os.path.exists(os.path.join(d, "fold4", "eval_probs.npy"))
    return all(ok(d) for d in runs) and (mix is None or ok(mix))


def main():
    ready = {}
    for name, (ev, runs, mix) in CV_SYSTEMS.items():
        if not have(runs, mix):
            print("skip (not downloaded yet):", name); continue
        out = os.path.join(PRED, name)
        run("cv_collect.py", "--eval", ev, "--run", *runs, *(["--mix", mix] if mix else []), "--out", out)
        ready[name] = os.path.join(out, "test_predictions.csv")
    for d in LLM_DIRS:
        p = os.path.join(d, "test_predictions.csv")
        if os.path.exists(p):
            ready["llm_" + os.path.basename(d)] = p
    print("systems ready:", sorted(ready))

    def evaluate(sub, names, baseline):
        names = [n for n in names if n in ready]
        if len(names) < 1:
            return
        base = baseline if baseline in names else names[0]
        run("evaluate_domain.py", "--pred", *[f"{n}={ready[n]}" for n in names], "--data", EV3, "--only-common",
            "--baseline", base, "--out", os.path.join(OUT, sub))   # --only-common: v4 runs add 58 Pidgin review texts

    evaluate("ablation", ABLATION, "tfidf_v2_setup")
    llms = [n for n in ready if n.startswith("llm_")]
    evaluate("main", MAIN + llms, "tfidf6_full")
    if llms:   # paired tests against the LLM with the best top-label accuracy
        def acc(n):
            rows = list(csv.DictReader(open(ready[n], encoding="utf-8-sig")))
            return sum(r["correct_primary"] == "True" for r in rows) / max(1, len(rows))
        best = max(llms, key=acc)
        evaluate("llm", [n for n in MAIN if n in ready] + llms, best)
    multi = os.path.join(HERE, "data_v3", "domain_multi_test.jsonl")
    if os.path.exists(multi):
        print("multi-domain test set present: score it with predict_domain_afroxlmr.py on the final package")


if __name__ == "__main__":
    main()

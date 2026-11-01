"""Run the rest of the v3 Kaggle plan unattended (Kaggle allows two GPU sessions at a time).

    PYTHONUTF8=1 python kaggle_v3/orchestrate.py [kernel-to-push-again ...]   # logs to results/kaggle_v3/orchestrate.log

Every 5 minutes:
  - download the output of any kernel that has finished into results/kaggle_v3/<kernel>/
  - once the Qwen kernel is done: turn its multi-domain candidate labels into data_v3/multi_silver.jsonl and review sheets
    (make_multi_candidates.py from-llm), build data_v3m (build_domain_data.py --relabel), add it to the Kaggle dataset as
    v3m_domain_{train,eval}.jsonl (new dataset version)
  - push the queued GPU kernels while a slot is free: Gemma first, then the final large and base models (these two only
    after data_v3m is on Kaggle)
Stops when everything has been pushed and has finished.
"""
import json, os, shutil, subprocess, sys, time

from kaggle.api.kaggle_api_extended import KaggleApi

K = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(K)
OUT = os.path.join(HERE, "results", "kaggle_v3")
LOG = os.path.join(OUT, "orchestrate.log")
USER = "bmsofficial"
WATCH = {"domain-v3-ablations": None, "domain-v3-tfidf": None, "domain-v3-llm-baselines": None,
         "domain-v3-llm-gemma": "k_llm_gemma", "domain-v3-final-large": "k_large", "domain-v3-final-base": "k_base_full"}
GPU = {"domain-v3-ablations", "domain-v3-llm-baselines", "domain-v3-llm-gemma", "domain-v3-final-large", "domain-v3-final-base"}
QUEUE = ["domain-v3-llm-gemma", "domain-v3-final-large", "domain-v3-final-base"]
NEEDS_V3M = {"domain-v3-final-large", "domain-v3-final-base"}

api = KaggleApi(); api.authenticate()
os.makedirs(OUT, exist_ok=True)


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M ") + " ".join(map(str, a))
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")


def status(k):
    try:
        return str(api.kernels_status(f"{USER}/{k}").status).split(".")[-1]
    except Exception:
        return "NONE"


def push(k):
    cwd = os.getcwd()
    os.chdir(os.path.join(K, WATCH[k]))
    try:
        r = api.kernels_push(".", acc="NvidiaTeslaT4")
        log("pushed", k, getattr(r, "url", None), getattr(r, "error", None) or "")
        return not getattr(r, "error", None)
    finally:
        os.chdir(cwd)


def download(k):
    d = os.path.join(OUT, k)
    os.makedirs(d, exist_ok=True)
    api.kernels_output(f"{USER}/{k}", d)
    log("downloaded", k, "->", d)


def build_v3m():
    py = sys.executable
    cand = os.path.join(OUT, "domain-v3-llm-baselines", "llm", "multi_candidates_qwen25_14b_6shot")
    relabel = []
    if os.path.exists(os.path.join(cand, "test_predictions.csv")):
        subprocess.run([py, "make_multi_candidates.py", "from-llm", "--data-dir", "data_v3", "--llm", cand], cwd=HERE, check=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        relabel = ["--relabel", os.path.join("data_v3", "multi_silver.jsonl")]
    else:
        log("no multi-domain candidate labels from Qwen; data_v3m = data_v3 without LLM rows")
    subprocess.run([py, "build_domain_data.py", "--out", "data_v3m", *relabel], cwd=HERE, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    ds = os.path.join(K, "dataset")
    for f in ("domain_model.py", "train_domain_classifier.py", "train_afroxlmr_domain.py", "llm_baseline_domain.py",
              "cv_collect.py", "evaluate_domain.py", "make_multi_candidates.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(ds, f))
    for s in ("train", "eval"):
        shutil.copy(os.path.join(HERE, "data_v3m", f"domain_{s}.jsonl"), os.path.join(ds, f"v3m_domain_{s}.jsonl"))
    cwd = os.getcwd(); os.chdir(ds)
    try:
        r = api.dataset_create_version(".", version_notes="add data_v3m (LLM multi-domain rows)", quiet=True, dir_mode="skip")
        log("dataset version", getattr(r, "status", r), getattr(r, "error", "") or "")
    finally:
        os.chdir(cwd)
    for _ in range(60):
        if "ready" in str(api.dataset_status(f"{USER}/domain-classifier-data-v3")).lower():
            break
        time.sleep(30)
    time.sleep(120)   # let the new version reach the kernel mounts
    log("data_v3m ready on Kaggle")


def main():
    redo = set(sys.argv[1:])   # kernels to push again even if an older version exists (e.g. domain-v3-llm-gemma)
    pushed = {k for k in QUEUE if status(k) != "NONE" and k not in redo}
    pushed_at = {k: 0.0 for k in pushed}
    fetched, failures, v3m = set(), {}, os.path.exists(os.path.join(K, "dataset", "v3m_domain_train.jsonl"))
    log("start; already pushed:", sorted(pushed), "| v3m ready:", v3m)
    while True:
        st = {k: status(k) for k in WATCH}
        for k, s in st.items():
            if s in ("COMPLETE", "ERROR") and k not in fetched and (k not in QUEUE or (k in pushed and time.time() - pushed_at[k] > 900)):
                log(k, s)
                try:
                    download(k)
                    fetched.add(k)
                except Exception as e:   # retried next round; after 3 failures move on without it
                    failures[k] = failures.get(k, 0) + 1
                    log("download failed", k, failures[k], e)
                    if failures[k] >= 3:
                        fetched.add(k)
        if not v3m and "domain-v3-llm-baselines" in fetched:
            build_v3m(); v3m = True
        busy = sum(st[k] in ("RUNNING", "QUEUED") for k in GPU)
        for k in QUEUE:
            if busy >= 2:
                break
            if k in pushed or (k in NEEDS_V3M and not v3m):
                continue
            if push(k):
                pushed.add(k); pushed_at[k] = time.time(); busy += 1
        if len(pushed) == len(QUEUE) and all(k in fetched for k in WATCH):
            log("all kernels finished"); return
        time.sleep(300)


if __name__ == "__main__":
    main()

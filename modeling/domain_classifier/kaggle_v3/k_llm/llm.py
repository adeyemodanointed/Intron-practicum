"""LLM baselines (Kaggle GPU T4 x2): Qwen 2.5 14B (4-bit), Qwen 2.5 7B and Gemma 3 12B (instruction-tuned), zero-shot
and 6-shot, on every human-labelled eval text. Qwen 2.5 14B 6-shot also labels the multi-domain candidates first
(multi_candidates.jsonl), so the final models do not wait for the rest.
Each model first answers 32 texts; if most answers come back empty (fp16 overflow), it is rerun 4-bit with float32 maths.
"""
import glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/common.py", recursive=True)[0]))
from common import SRC, W, done, sh  # noqa: E402

import json, subprocess  # noqa: E402
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)


CONFIGS = [p.replace("\\", "/") for p in glob.glob("/kaggle/input/**/config.json", recursive=True)]
print("attached model folders:", CONFIGS, flush=True)


def find(slug):
    """The attached model folder for a Kaggle model instance slug (mount paths differ between Kaggle images)."""
    hits = sorted(os.path.dirname(p) for p in CONFIGS if f"/{slug}/" in p)
    return hits[-1] if hits else None


MODELS = [("qwen25_14b", find("14b-instruct"), 16, ["--load-4bit"]),
          ("qwen25_7b", find("7b-instruct"), 16, []),
          ("gemma3_12b", find("gemma-3-12b-it"), 8, [])]
print("models", MODELS, flush=True)
L = "llm_baseline_domain.py"
COMMON = ["--backend", "hf", "--input", "data_v3/domain_eval.jsonl", "--train", "data_v3/domain_train.jsonl"]
status = {}


def run_model(name, path, batch, extra):
    if not path:
        status[name] = "model not attached"; return
    sh(L, *COMMON, "--model", path, "--batch", batch, *extra, "--limit", 32, "--out", f"smoke/{name}")
    answers = [json.loads(l)["answer"].strip() for l in open(f"{W}/smoke/{name}/responses.jsonl", encoding="utf-8")]
    if sum(not a for a in answers) > len(answers) / 2:
        extra = ["--load-4bit", "--dtype", "float32"]
    status[name] = {"extra": extra, "smoke_empty": sum(not a for a in answers)}
    if name == "qwen25_14b":
        sh(L, "--backend", "hf", "--model", path, "--batch", batch, *extra, "--shots", 6, "--train", "data_v3/domain_train.jsonl",
           "--input", f"{SRC}/multi_candidates.jsonl", "--all-rows", "--out", f"llm/multi_candidates_{name}_6shot")
    for shots in (0, 6):
        sh(L, *COMMON, "--model", path, "--batch", batch, *extra, "--shots", shots, "--out", f"llm/{name}_{shots}shot")


for name, path, batch, extra in MODELS:
    try:   # one model failing (e.g. out of memory) must not lose the others' results
        run_model(name, path, batch, extra)
    except subprocess.CalledProcessError as e:
        status[name] = f"failed: {e}"
        print("FAILED", name, e, flush=True)
shutil.rmtree(f"{W}/smoke", ignore_errors=True)
done(status=status)

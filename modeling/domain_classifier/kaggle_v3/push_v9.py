"""Copy data_v9_clean into the small v9 dataset, create it on Kaggle (retrying on network errors), then push the v9
AfroXLMR-base kernel (GPU T4)."""
import os, shutil, time
from kaggle.api.kaggle_api_extended import KaggleApi
K = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(K), "data_v9_clean")
for s in ("train", "eval"):
    shutil.copy(os.path.join(SRC, f"domain_{s}.jsonl"), os.path.join(K, "dataset_v9", f"v9_domain_{s}.jsonl"))
api = KaggleApi(); api.authenticate()
for attempt in range(6):
    try:
        os.chdir(os.path.join(K, "dataset_v9"))
        r = api.dataset_create_new(".", public=False, quiet=True, dir_mode="skip")
        print("dataset", getattr(r, "status", r), getattr(r, "error", "") or "", flush=True)
        break
    except Exception as e:
        print("upload failed, retrying in 60 s:", str(e)[:150], flush=True); time.sleep(60)
for _ in range(60):
    try:
        if "ready" in str(api.dataset_status("bmsofficial/domain-classifier-data-v9")).lower():
            break
    except Exception:
        pass
    time.sleep(20)
time.sleep(90)
for attempt in range(6):
    try:
        os.chdir(os.path.join(K, "k_v9"))
        r = api.kernels_push(".", acc="NvidiaTeslaT4")
        print("push k_v9", getattr(r, "url", None), getattr(r, "error", None) or "", flush=True)
        break
    except Exception as e:
        print("push failed, retrying in 60 s:", str(e)[:150], flush=True); time.sleep(60)

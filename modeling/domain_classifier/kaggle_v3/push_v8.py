"""Create the small v8 dataset (retrying on network errors), then push the v8 base (GPU) and TF-IDF (CPU) kernels."""
import os, time
from kaggle.api.kaggle_api_extended import KaggleApi
K = os.path.dirname(os.path.abspath(__file__))
api = KaggleApi(); api.authenticate()
for attempt in range(6):
    try:
        os.chdir(os.path.join(K, "dataset_v8"))
        r = api.dataset_create_new(".", public=False, quiet=True, dir_mode="skip")
        print("dataset", getattr(r, "status", r), getattr(r, "error", "") or "", flush=True)
        break
    except Exception as e:
        print("upload failed, retrying in 60 s:", str(e)[:150], flush=True); time.sleep(60)
for _ in range(60):
    try:
        if "ready" in str(api.dataset_status("bmsofficial/domain-classifier-data-v8")).lower():
            break
    except Exception:
        pass
    time.sleep(20)
time.sleep(90)
for k, acc in (("k_v8", "NvidiaTeslaT4"), ("k_tfidf_v8", None)):
    os.chdir(os.path.join(K, k))
    r = api.kernels_push(".", acc=acc) if acc else api.kernels_push(".")
    print("push", k, getattr(r, "url", None), getattr(r, "error", None) or "", flush=True)

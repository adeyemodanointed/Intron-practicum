"""Push a kernel folder as soon as fewer than two of our GPU kernels are running, then wait for it and download its
output: python kaggle_v3/push_when_free.py k_v5 domain-v5-base"""
import os, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi

folder, slug = sys.argv[1], sys.argv[2]
K = os.path.dirname(os.path.abspath(__file__))
GPU = ["domain-v3-final-large", "domain-v4-base", "domain-v5-base", "domain-v3-final-base", "domain-v3-ablations"]
api = KaggleApi(); api.authenticate()


def status(k):
    try:
        return str(api.kernels_status(f"bmsofficial/{k}").status).split(".")[-1]
    except Exception:
        return "NONE"


while sum(status(k) in ("RUNNING", "QUEUED") for k in GPU if k != slug) >= 2:
    time.sleep(300)
os.chdir(os.path.join(K, folder))
r = api.kernels_push(".", acc="NvidiaTeslaT4")
print("pushed", slug, getattr(r, "url", None), getattr(r, "error", None) or "", time.strftime("%H:%M"), flush=True)
time.sleep(600)
while status(slug) not in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
    time.sleep(300)
d = os.path.join(os.path.dirname(K), "results", "kaggle_v3", slug)
os.makedirs(d, exist_ok=True)
api.kernels_output(f"bmsofficial/{slug}", d)
print(slug, status(slug), "downloaded to", d, time.strftime("%H:%M"), flush=True)

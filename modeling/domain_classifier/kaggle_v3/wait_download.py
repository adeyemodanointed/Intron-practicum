"""Wait for Kaggle kernels to finish, then download their output; survives network drops (retries every few minutes).
    python kaggle_v3/wait_download.py <kernel-slug> [<kernel-slug> ...]"""
import os, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "kaggle_v3")


def retry(f, *a):
    while True:
        try:
            return f(*a)
        except Exception as e:   # no internet / DNS failure / API hiccup: wait and try again
            print(time.strftime("%H:%M"), "network error, retrying in 3 min:", str(e)[:100], flush=True)
            time.sleep(180)


api = KaggleApi(); retry(api.authenticate)
for k in sys.argv[1:]:
    while True:
        s = str(retry(api.kernels_status, f"bmsofficial/{k}").status).split(".")[-1]
        if s in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            break
        time.sleep(300)
    d = os.path.join(OUT, k); os.makedirs(d, exist_ok=True)
    retry(api.kernels_output, f"bmsofficial/{k}", d)
    print(time.strftime("%H:%M"), k, s, "downloaded to", d, flush=True)

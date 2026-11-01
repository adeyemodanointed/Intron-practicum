"""Wait for the running domain-v3-teamtest version, push the updated one (adds TF-IDF), wait, download its output."""
import os, time
from kaggle.api.kaggle_api_extended import KaggleApi
K = os.path.dirname(os.path.abspath(__file__))
api = KaggleApi(); api.authenticate()
st = lambda: str(api.kernels_status("bmsofficial/domain-v3-teamtest").status).split(".")[-1]
while st() in ("RUNNING", "QUEUED"):
    time.sleep(120)
os.chdir(os.path.join(K, "k_teamtest")); r = api.kernels_push("."); print("pushed", getattr(r, "error", None) or "ok", flush=True)
time.sleep(300)
while st() in ("RUNNING", "QUEUED"):
    time.sleep(120)
d = os.path.join(os.path.dirname(K), "results", "kaggle_v3", "domain-v3-teamtest"); os.makedirs(d, exist_ok=True)
api.kernels_output("bmsofficial/domain-v3-teamtest", d); print("status", st(), "downloaded", d, flush=True)

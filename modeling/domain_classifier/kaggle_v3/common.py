"""Shared setup for the v3 Kaggle kernels: find the dataset, lay out data_v3/ and data_v3c/ in /kaggle/working."""
import glob, json, os, subprocess, sys, time

SRC = os.path.dirname(glob.glob("/kaggle/input/**/train_afroxlmr_domain.py", recursive=True)[0])
W = "/kaggle/working"
T0 = time.time()
for v in ("v3", "v3c", "v3m", "v4", "v5", "v6", "v7", "v8", "v9"):
    if os.path.exists(f"{SRC}/{v}_domain_train.jsonl"):
        os.makedirs(f"{W}/data_{v}", exist_ok=True)
        for s in ("train", "eval"):
            dst = f"{W}/data_{v}/domain_{s}.jsonl"
            if not os.path.exists(dst):
                os.symlink(f"{SRC}/{v}_domain_{s}.jsonl", dst)


def sh(script, *args):
    cmd = [sys.executable, f"{SRC}/{script}", *map(str, args)]
    print(f"\n===== [{(time.time() - T0) / 60:.0f} min] {script} {' '.join(map(str, args))}", flush=True)
    subprocess.run(cmd, check=True, cwd=W)


def done(**info):
    json.dump({"minutes": round((time.time() - T0) / 60), **info}, open(f"{W}/done.json", "w"), indent=1)
    print("DONE", f"{(time.time() - T0) / 60:.0f} min", info, flush=True)

"""Overnight AfroXLMR tuning run on Kaggle (GPU T4), all on the private dataset domain-classifier-data-v2.

1. Learning-rate sweep: 1e-5, 2e-5, 3e-5 on 20,000 training rows, 2 epochs, dev checked every half epoch.
   Dev only; the test set is not touched.
2. Full runs with the best learning rate: up to 15,000 rows per language and label (about 95,000 rows),
   up to 4 epochs, dev checked every half epoch, best checkpoint kept; three seeds.
3. Average the three seeds (thresholds tuned on dev) and score the test set once.
Everything is written to /kaggle/working (kept as this notebook's output).
"""
import glob, json, os, subprocess, sys, time

SRC = os.path.dirname(glob.glob("/kaggle/input/**/train_afroxlmr_domain.py", recursive=True)[0])
W = "/kaggle/working"
T0 = time.time()


def run(args, out):
    cmd = [sys.executable, f"{SRC}/train_afroxlmr_domain.py", "--data-dir", SRC, "--out-dir", out] + args
    print(f"\n===== [{(time.time() - T0) / 60:.0f} min] {' '.join(cmd[2:])}", flush=True)
    subprocess.run(cmd, check=True)
    return json.load(open(f"{out}/summary.json"))


run(["--max-train", "300", "--max-eval", "200", "--epochs", "1", "--eval-every", "0.5"], f"{W}/smoke")   # fails fast if anything is broken
subprocess.run(["rm", "-rf", f"{W}/smoke"])

sweep = {}
for lr in ("1e-5", "2e-5", "3e-5"):
    s = run(["--lr", lr, "--epochs", "2", "--eval-every", "0.5", "--warmup", "0.1", "--cap-per-cell", "15000",
             "--max-train", "20000", "--no-test"], f"{W}/sweep/lr{lr}")
    sweep[lr] = s["best_dev_mean_macro_f1"]
    print("SWEEP", lr, s["best_dev_mean_macro_f1"], "at", s["best_at"], flush=True)
best_lr = max(sweep, key=sweep.get)
json.dump({"sweep_dev_mean_macro_f1": sweep, "best_lr": best_lr}, open(f"{W}/sweep/sweep_summary.json", "w"), indent=1)
print("BEST LR", best_lr, sweep, flush=True)

runs = []
for seed in (13, 14, 15):
    out = f"{W}/runs/seed{seed}"
    s = run(["--lr", best_lr, "--epochs", "4", "--eval-every", "0.5", "--warmup", "0.1", "--cap-per-cell", "15000",
             "--seed", str(seed)], out)
    print("RUN seed", seed, s["best_dev_mean_macro_f1"], "at", s["best_at"], flush=True)
    runs.append(out)

subprocess.run([sys.executable, f"{SRC}/ensemble_domain.py", "--data-dir", SRC, "--runs", *runs, "--out", f"{W}/ensemble"], check=True)
best = max(runs, key=lambda r: json.load(open(f"{r}/summary.json"))["best_dev_mean_macro_f1"])
json.dump({"best_seed_dir": best, "minutes": round((time.time() - T0) / 60)}, open(f"{W}/done.json", "w"), indent=1)
print("DONE", best, f"{(time.time() - T0) / 60:.0f} min", flush=True)

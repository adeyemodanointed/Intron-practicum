"""v4 data (unused sources added: DigitalUmuganda Kinyarwanda health text, SIB-200, Afrivoice Swahili passages, more
KINNEWS categories, LLM multi-domain rows) with AfroXLMR-base, same folds and recipe as the v3 runs (Kaggle GPU T4).
  v4_multi6      six outputs, stage 1 + stage 2 (the base_d recipe)
  v4_multi6_mix  the same plus two-label mixing (10%)
"""
import glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/common.py", recursive=True)[0]))
from common import SRC, W, done, sh  # noqa: E402

BASE = "Davlan/afro-xlmr-base"
S1 = ["--stage", 1, "--model", BASE, "--epochs", 2, "--lr", "2e-5", "--cap-per-cell", 15000, "--batch", 16, "--head", "multi6"]
S2 = ["--stage", 2, "--epochs", 4, "--lr", "2e-5", "--eval-every", 0.5, "--patience", 3, "--replay-per-cell", 200,
      "--gold-share", 0.6, "--warmup", 0.1, "--head", "multi6"]
T = "train_afroxlmr_domain.py"

sh(T, "--data-dir", "data_v4", "--out-dir", "smoke/s1", *S1, "--mix-rate", 0, "--max-train", 200, "--max-eval", 300, "--epochs", 0.3)
shutil.rmtree(f"{W}/smoke")
for name, mix in [("v4_multi6", 0), ("v4_multi6_mix", 0.1)]:
    s1 = f"runs/{name}/stage1"
    sh(T, "--data-dir", "data_v4", "--out-dir", s1, *S1, "--mix-rate", mix)
    for k in range(5):
        out = f"runs/{name}/stage2/fold{k}"
        sh(T, "--data-dir", "data_v4", "--out-dir", out, *S2, "--mix-rate", mix, "--init-from", f"{s1}/model", "--fold", k)
        shutil.rmtree(f"{W}/{out}/model")
    shutil.rmtree(f"{W}/{s1}/model")
done(runs=["v4_multi6", "v4_multi6_mix"])

"""AfroXLMR-base on v9 data (Kaggle GPU T4): v8 plus FineWeb-2 Pidgin and Yoruba web text (mostly real off-topic
`other` texts in the style of the team-validated ones), then label cleaning (clean_labels.py: weak/teacher rows whose
out-of-fold TF-IDF scores contradict their label are dropped). Same settings as v8, so the two compare directly.
"""
import glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/common.py", recursive=True)[0]))
from common import SRC, W, done, sh  # noqa: E402

BASE = "Davlan/afro-xlmr-base"
COMMON = ["--head", "multi6", "--mix-rate", 0.1, "--batch", 16]
S1 = ["--stage", 1, "--model", BASE, "--epochs", 2, "--lr", "2e-5", "--cap-per-cell", 10000, *COMMON]
S2 = ["--stage", 2, "--epochs", 4, "--lr", "2e-5", "--eval-every", 0.5, "--patience", 2, "--replay-per-cell", 200,
      "--gold-share", 0.6, "--warmup", 0.1, *COMMON]
T = "train_afroxlmr_domain.py"
D = "data_v9"

sh(T, "--data-dir", D, "--out-dir", "smoke/s1", *S1, "--max-train", 200, "--max-eval", 300, "--epochs", 0.3)
shutil.rmtree(f"{W}/smoke")
s1 = "runs/v9/stage1"
sh(T, "--data-dir", D, "--out-dir", s1, *S1)
for k in [0, 1, 2, 3, 4, -1]:
    out = f"runs/v9/stage2/fold{k}" if k >= 0 else "runs/v9/final"
    sh(T, "--data-dir", D, "--out-dir", out, *S2, "--init-from", f"{s1}/model", "--fold", k)
    if k >= 0:
        shutil.rmtree(f"{W}/{out}/model")
shutil.rmtree(f"{W}/{s1}/model")
done(run="v9")

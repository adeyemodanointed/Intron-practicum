"""v3 final AfroXLMR-large (Kaggle GPU T4): data_v3m (KINNEWS + MasakhaNEWS + LLM multi-domain rows), six outputs,
two-label mixing (10% of rows), stage 1 then stage 2 on the human labels of each fold, plus the final model
(stage 2 on folds 1-4, dev fold 0; weights kept in runs/large/final/model). Scored with cv_collect.py / evaluate_domain.py.
"""
import glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/common.py", recursive=True)[0]))
from common import SRC, W, done, sh  # noqa: E402

COMMON = ["--head", "multi6", "--mix-rate", 0.1, "--batch", 8, "--grad-accum", 2, "--grad-checkpointing"]
S1 = ["--stage", 1, "--model", "Davlan/afro-xlmr-large", "--epochs", 2, "--lr", "1e-5", "--cap-per-cell", 8000, *COMMON]
S2 = ["--stage", 2, "--epochs", 4, "--lr", "1e-5", "--eval-every", 0.5, "--patience", 3, "--replay-per-cell", 200,
      "--gold-share", 0.6, "--warmup", 0.1, *COMMON]
T = "train_afroxlmr_domain.py"
D = "data_v3m"

sh(T, "--data-dir", D, "--out-dir", "smoke/s1", *S1, "--max-train", 200, "--max-eval", 300, "--epochs", 0.3)   # fails fast
sh(T, "--data-dir", D, "--out-dir", "smoke/s2", *S2, "--init-from", "smoke/s1/model", "--fold", 0, "--max-train", 200, "--max-eval", 300, "--epochs", 0.5)
shutil.rmtree(f"{W}/smoke")

s1 = "runs/large/stage1"
sh(T, "--data-dir", D, "--out-dir", s1, *S1)
for k in [0, 1, 2, 3, 4, -1]:
    out = f"runs/large/stage2/fold{k}" if k >= 0 else "runs/large/final"
    sh(T, "--data-dir", D, "--out-dir", out, *S2, "--init-from", f"{s1}/model", "--fold", k)
    if k >= 0:
        shutil.rmtree(f"{W}/{out}/model")
shutil.rmtree(f"{W}/{s1}/model")
done(run="large")

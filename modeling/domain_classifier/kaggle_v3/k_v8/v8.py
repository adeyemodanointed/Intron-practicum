"""Final AfroXLMR-base on v8 data (Kaggle GPU T4): no LLM-generated text, real Yoruba (SIB-200, WURA), Afrivoice
Kinyarwanda transcripts, and the team's manually validated texts in the cross-validation pool. Six outputs, two-label
mixing (10%), stage 1 then stage 2 on each fold (out-of-fold scores for every human-labelled text) and the final model
(folds 1-4, dev fold 0; weights kept in runs/v8/final/model).
"""
import glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/common.py", recursive=True)[0]))
from common import SRC, W, done, sh  # noqa: E402

BASE = "Davlan/afro-xlmr-base"
COMMON = ["--head", "multi6", "--mix-rate", 0.1, "--batch", 16]
# GPU quota is short: 10,000 rows per language and label in stage 1 (was 15,000) and early stop after 2 checks
S1 = ["--stage", 1, "--model", BASE, "--epochs", 2, "--lr", "2e-5", "--cap-per-cell", 10000, *COMMON]
S2 = ["--stage", 2, "--epochs", 4, "--lr", "2e-5", "--eval-every", 0.5, "--patience", 2, "--replay-per-cell", 200,
      "--gold-share", 0.6, "--warmup", 0.1, *COMMON]
T = "train_afroxlmr_domain.py"
D = "data_v8"

sh(T, "--data-dir", D, "--out-dir", "smoke/s1", *S1, "--max-train", 200, "--max-eval", 300, "--epochs", 0.3)
shutil.rmtree(f"{W}/smoke")
s1 = "runs/v8/stage1"
sh(T, "--data-dir", D, "--out-dir", s1, *S1)
for k in [0, 1, 2, 3, 4, -1]:
    out = f"runs/v8/stage2/fold{k}" if k >= 0 else "runs/v8/final"
    sh(T, "--data-dir", D, "--out-dir", out, *S2, "--init-from", f"{s1}/model", "--fold", k)
    if k >= 0:
        shutil.rmtree(f"{W}/{out}/model")
shutil.rmtree(f"{W}/{s1}/model")
done(run="v8")

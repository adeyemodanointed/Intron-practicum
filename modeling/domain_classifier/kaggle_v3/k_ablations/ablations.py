"""v3 ablations with AfroXLMR-base (Kaggle GPU T4), 5-fold cross-validation on the human-labelled texts.

  c_sigmoid5  commercial-safe data, five outputs (other = nothing passed): stage 1 = the v2 setup; stage 2 adds human labels
  c_multi6    commercial-safe data, six outputs (other is a real output), stage 1 + stage 2
  v3_multi6   + KINNEWS and MasakhaNEWS swa/yor, six outputs, stage 1 + stage 2
Probabilities for every eval row are kept (runs/<name>/stage1/eval_probs.npy, runs/<name>/stage2/fold<k>/eval_probs.npy);
model weights are deleted to keep the output small. Scored afterwards with cv_collect.py and evaluate_domain.py.
"""
import glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/common.py", recursive=True)[0]))
from common import SRC, W, done, sh  # noqa: E402

BASE = "Davlan/afro-xlmr-base"
S1 = ["--stage", 1, "--model", BASE, "--epochs", 2, "--lr", "2e-5", "--cap-per-cell", 15000, "--batch", 16, "--mix-rate", 0]
S2 = ["--stage", 2, "--epochs", 4, "--lr", "2e-5", "--eval-every", 0.5, "--patience", 3, "--replay-per-cell", 200,
      "--gold-share", 0.6, "--warmup", 0.1, "--mix-rate", 0]
T = "train_afroxlmr_domain.py"

sh(T, "--data-dir", "data_v3", "--out-dir", "smoke/s1", *S1, "--max-train", 200, "--max-eval", 300, "--epochs", 0.3)   # fails fast
sh(T, "--data-dir", "data_v3", "--out-dir", "smoke/s2", *S2, "--init-from", "smoke/s1/model", "--fold", 0, "--max-train", 200, "--max-eval", 300, "--epochs", 0.5)
shutil.rmtree(f"{W}/smoke")

for name, data, head in [("c_sigmoid5", "data_v3c", "sigmoid5"), ("c_multi6", "data_v3c", "multi6"), ("v3_multi6", "data_v3", "multi6")]:
    s1 = f"runs/{name}/stage1"
    sh(T, "--data-dir", data, "--out-dir", s1, "--head", head, *S1)
    for k in range(5):
        out = f"runs/{name}/stage2/fold{k}"
        sh(T, "--data-dir", data, "--out-dir", out, "--head", head, *S2, "--init-from", f"{s1}/model", "--fold", k)
        shutil.rmtree(f"{W}/{out}/model")
    shutil.rmtree(f"{W}/{s1}/model")
done(runs=["c_sigmoid5", "c_multi6", "v3_multi6"])

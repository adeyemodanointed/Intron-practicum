"""v3 TF-IDF baselines (Kaggle CPU), the same 5 folds as the AfroXLMR runs.

  cv/tfidf6     KINNEWS + MasakhaNEWS data, six outputs, human labels of the training folds (gold share 0.15/0.3/0.5 chosen
                on each dev fold) + the final model (folds 1-4, dev fold 0) saved as cv/tfidf6/final/domain_classifier.joblib
  cv/tfidf5_c   commercial-safe data, five outputs, with human labels (stage 1 = the v2 setup)
"""
import glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/common.py", recursive=True)[0]))
from common import SRC, W, done, sh  # noqa: E402

sh("train_domain_classifier.py", "--data-dir", "data_v3", "--cv", "cv/tfidf6", "--other-output", "--gold-share", 0.15, 0.3, 0.5, "--folds", 0, 1, 2, 3, 4, -1)
sh("train_domain_classifier.py", "--data-dir", "data_v3c", "--cv", "cv/tfidf5_c", "--gold-share", 0.15, 0.3, 0.5)
done(runs=["tfidf6", "tfidf5_c"])

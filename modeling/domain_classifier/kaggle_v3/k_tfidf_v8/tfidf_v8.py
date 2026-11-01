"""TF-IDF on v8 data (Kaggle CPU), the same folds as domain-v8-base: six outputs, human labels of the training folds
(gold share 0.15/0.3/0.5 chosen per dev fold), plus the final model (folds 1-4, dev fold 0)."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/common.py", recursive=True)[0]))
from common import done, sh  # noqa: E402

sh("train_domain_classifier.py", "--data-dir", "data_v8", "--cv", "cv/tfidf6_v8", "--other-output", "--gold-share", 0.15, 0.3, 0.5,
   "--folds", 0, 1, 2, 3, 4, -1)
done(runs=["tfidf6_v8"])

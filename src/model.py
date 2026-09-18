"""Fixed model configuration shared by every ROAR condition.

Hyperparameters are frozen and identical across baseline, informed-removal
and random-removal conditions on purpose: any PR-AUC change must reflect the
information content of the removed features, not per-condition re-tuning.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

N_ESTIMATORS = 150
MAX_DEPTH = 3
LEARNING_RATE = 0.1
SUBSAMPLE = 0.9
COLSAMPLE_BYTREE = 0.9

REPEAT_SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]  # 10 repeats
N_SPLITS = 5


def make_model(monotone_constraints, seed):
    return XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        monotone_constraints=monotone_constraints,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )


def cv_pr_auc(X, y, monotone_constraints, repeat_seeds=REPEAT_SEEDS, n_splits=N_SPLITS):
    """Repeated stratified k-fold PR-AUC (average precision).

    Returns the array of per-fold AP scores (len == len(repeat_seeds) * n_splits)
    so callers can report mean/SD or build percentile distributions.
    """
    scores = []
    for r_i, seed in enumerate(repeat_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            model_seed = seed * 100 + fold_i
            model = make_model(monotone_constraints, model_seed)
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            scores.append(average_precision_score(y[test_idx], proba))
    return np.array(scores)


def fit_full(X, y, monotone_constraints, seed=2026):
    model = make_model(monotone_constraints, seed)
    model.fit(X, y)
    return model

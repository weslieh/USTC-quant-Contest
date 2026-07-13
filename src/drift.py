"""Adversarial-validation drift ranking for feature dropping.

Train a LightGBM regressor (0/1 labels) to distinguish train rows from test
rows; features with high importance in that classifier are the ones whose
distribution shifts most between train and test. Dropping them removes noise
on the shifted test distribution.

The rank is computed once (over a sample for speed) and reused across folds.
"""

import numpy as np
import polars as pl
from pathlib import Path
from lightgbm import LGBMRegressor
import lightgbm as lgb


def compute_drift_rank(
    train_paths,
    test_paths,
    feature_cols,
    n_sample_train=600_000,
    n_sample_test=400_000,
    seed=42,
):
    """Return feature names sorted by descending train-vs-test drift importance.

    ``train_paths``/``test_paths`` are parquet path lists (or single paths).
    Uses a sample to keep it fast; AUC≈1.0 means train/test are fully separable.
    """
    if isinstance(train_paths, (str, Path)):
        train_paths = [train_paths]
    if isinstance(test_paths, (str, Path)):
        test_paths = [test_paths]

    # Sample evenly across the provided parquet files so the drift sample spans
    # all of them (a single .head() on a multi-file scan only reads the first).
    per_file = max(1, n_sample_train // max(1, len(train_paths)))
    tr_frames = [pl.scan_parquet(p).head(per_file).select(feature_cols).collect()
                 for p in train_paths]
    tr = pl.concat(tr_frames, how="vertical_relaxed")
    te = pl.scan_parquet(test_paths).head(n_sample_test).select(feature_cols).collect()

    Xtr = np.nan_to_num(tr.to_numpy().astype(np.float32))
    Xte = np.nan_to_num(te.to_numpy().astype(np.float32))
    n = min(len(Xtr), len(Xte))
    X = np.vstack([Xtr[:n], Xte[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    sp = int(0.7 * len(X))

    m = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=-1, verbosity=-1,
    )
    m.fit(X[:sp], y[:sp], eval_set=[(X[sp:], y[sp:])],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    p = m.predict(X[sp:])
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[sp:], p)
    except Exception:
        auc = float("nan")

    imp = dict(zip(feature_cols, m.feature_importances_.tolist()))
    ranked = sorted(imp, key=imp.get, reverse=True)
    print(f"  drift AUC(train vs test)={auc:.4f}; top-5 drift: {ranked[:5]}", flush=True)
    return ranked


def drop_drift_features(raw_feature_cols, drift_rank, k):
    """Return raw_feature_cols with the top-k drift features removed."""
    if k <= 0:
        return list(raw_feature_cols)
    drop = set(drift_rank[:k])
    kept = [c for c in raw_feature_cols if c not in drop]
    print(f"  dropped {len(raw_feature_cols) - len(kept)} drift features (k={k}); "
          f"{len(kept)} remain", flush=True)
    return kept

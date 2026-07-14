"""Adversarial-validation fold selection using real test features.

The earlier version of this module used a *pseudo-test* set (late vs early
train) as the adversarial target. With train-vs-test AUC=1.0, a late-vs-early
classifier is also near-perfect and produces degenerate ``p_like_test`` scores
(threshold saturates at 1.0), so the "most test-like" selection became
arbitrary. This rewrite trains the adversarial classifier against the **real
test features** (which are available in the public release) so the
"test-likeness" score is meaningful, then builds time-respecting folds whose
validation rows are the most test-like among those that come *after* the
training cutoff (no temporal leakage).

The same per-row ``p_like_test`` is also reused for covariate-shift sample
reweighting — see src/reweight.py.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl


def _resolve_paths(data_root, split):
    """Return the list of parquet paths for ``split`` via the manifest."""
    from src.dataset import _partition_paths

    return _partition_paths(data_root, split)


def train_adversarial_classifier(
    train_paths,
    test_paths,
    raw_feature_cols,
    n_sample_train=600_000,
    n_sample_test=400_000,
    seed=42,
):
    """Train a LightGBM binary classifier to tell train (0) from test (1) rows.

    Samples evenly across the provided parquet files (a single ``.head()`` on a
    multi-file scan only reads the first file). Returns the fitted classifier.

    ``raw_feature_cols`` must be present in both train and test schemas (the
    323 ``feature_*`` columns are identical across splits).
    """
    if isinstance(train_paths, (str, Path)):
        train_paths = [train_paths]
    if isinstance(test_paths, (str, Path)):
        test_paths = [test_paths]

    per_file = max(1, n_sample_train // max(1, len(train_paths)))
    tr_frames = [
        pl.scan_parquet(p).head(per_file).select(raw_feature_cols).collect()
        for p in train_paths
    ]
    tr = pl.concat(tr_frames, how="vertical_relaxed")
    te = pl.scan_parquet(test_paths).head(n_sample_test).select(raw_feature_cols).collect()

    Xtr = np.nan_to_num(tr.to_numpy().astype(np.float32))
    Xte = np.nan_to_num(te.to_numpy().astype(np.float32))
    n = min(len(Xtr), len(Xte))
    X = np.vstack([Xtr[:n], Xte[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    sp = int(0.7 * len(X))

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(X[:sp], y[:sp])

    # Report the real train-vs-test AUC (not the pseudo-test one).
    from sklearn.metrics import roc_auc_score

    p = clf.predict_proba(X[sp:])[:, 1]
    try:
        auc = float(roc_auc_score(y[sp:], p))
    except Exception:
        auc = float("nan")
    print(f"  [AV] train-vs-test AUC={auc:.4f}", flush=True)
    return clf


def score_rows(clf, X: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """Predict ``p_like_test`` for every row of ``X`` in chunks."""
    out = np.empty(X.shape[0], dtype=np.float32)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        out[s:e] = clf.predict_proba(X[s:e])[:, 1]
    return out


def compute_time_adv_scores(
    df: pl.LazyFrame,
    raw_feature_cols: list[str],
    clf,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every train row with ``clf`` and aggregate to per-time_id mean.

    Returns ``(sorted_time_ids, adv_score_per_time_id)`` where the arrays are
    sorted ascending by ``time_id``. The per-row score is also returned via the
    side channel ``df``'s collected frame is not stored here; callers that need
    per-row scores (reweighting) should call ``score_rows`` directly on the
    collected feature matrix.
    """
    full = df.collect()
    X = np.nan_to_num(full.select(raw_feature_cols).to_numpy().astype(np.float32))
    row_scores = score_rows(clf, X)
    full = full.with_columns(pl.Series("_adv_score", row_scores))
    per = (
        full.group_by("time_id")
        .agg(pl.col("_adv_score").mean().alias("adv_score"))
        .sort("time_id")
    )
    times = per["time_id"].to_numpy()
    scores = per["adv_score"].to_numpy().astype(np.float32)
    return times, scores


def compute_score_by_time(
    df: pl.LazyFrame,
    raw_feature_cols: list[str],
    data_root: str = "data",
    sample_rows: int = 1_000_000,
    seed: int = 42,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Train the AV classifier (real test) once and return per-time_id scores.

    Returns ``(score_by_time, sorted_time_ids, adv_scores)`` where
    ``score_by_time`` maps ``time_id -> p_like_test`` (mean across assets in
    that time_id). Reused by both validation-set selection and covariate-shift
    reweighting so the classifier is trained at most once per run.
    """
    train_paths = _resolve_paths(data_root, "train")
    test_paths = _resolve_paths(data_root, "test")
    if not test_paths:
        return {}, np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    print(f"  [AV] training adversarial classifier (real test features) ...")
    clf = train_adversarial_classifier(
        train_paths,
        test_paths,
        raw_feature_cols,
        n_sample_train=min(sample_rows, 600_000),
        n_sample_test=400_000,
        seed=seed,
    )
    print("  [AV] scoring full train panel for test-likeness ...")
    times, adv_scores = compute_time_adv_scores(df, raw_feature_cols, clf)
    score_by_time = {int(t): float(s) for t, s in zip(times, adv_scores)}
    print(
        f"  [AV] {len(times)} time_ids; "
        f"adv_score range [{adv_scores.min():.4f}, {adv_scores.max():.4f}]"
    )
    return score_by_time, times, adv_scores


def build_folds_from_scores(
    df: pl.LazyFrame,
    times: np.ndarray,
    adv_scores: np.ndarray,
    adv_val_ratio: float = 0.1,
    n_folds: int = 5,
    embargo: int = 0,
):
    """Build time-respecting folds whose validation rows are most test-like.

    Expanding-window structure mirrors ``time_cv_split``: each fold trains on
    more history. Within each fold's eligible region (ALL time_ids after the
    training cutoff), the ``valid_size`` time_ids with the highest adversarial
    test-likeness are chosen as validation. This keeps the temporal ordering
    (validation is always after training) while making the validation set
    resemble the test distribution.

    Returns a list of ``(train_lf, valid_lf)`` lazy frames.
    """
    from src.cv import time_cv_split

    n_total = len(times)
    valid_size = max(1, int(n_total * adv_val_ratio))
    score_by_time = {int(t): float(s) for t, s in zip(times, adv_scores)}

    folds = []
    for f in range(n_folds):
        # Expanding-window training cutoff: each fold trains on more history.
        # The last fold trains on everything except the final valid window.
        train_cutoff = n_total - (n_folds - f) * valid_size
        if f == n_folds - 1:
            train_cutoff = n_total - valid_size  # final fold: leave a tail
        train_end = train_cutoff - embargo  # embargo gap before eligible region
        if train_end <= 0:
            continue

        # Eligible region = ALL time_ids after the training cutoff (the whole
        # future), so we can genuinely prefer the most test-like ones while
        # keeping the temporal ordering (every valid time_id > every train one).
        eligible_idx = np.arange(train_cutoff, n_total)
        if eligible_idx.size < valid_size:
            continue
        elig_scores = adv_scores[eligible_idx]
        # Top-valid_size most test-like time_ids within the eligible future.
        top_local = np.argpartition(elig_scores, -valid_size)[-valid_size:]
        chosen = np.sort(eligible_idx[top_local])  # keep time order within valid
        valid_times = times[chosen]
        train_times = times[:train_end]

        train_lf = df.filter(pl.col("time_id").is_in(train_times.tolist()))
        valid_lf = df.filter(pl.col("time_id").is_in(valid_times.tolist()))
        mean_score = float(np.mean([score_by_time[int(t)] for t in valid_times]))
        print(
            f"  [AV] fold {f}: train time_ids {train_times[0]}..{train_times[-1]} "
            f"({len(train_times)}), valid {valid_times[0]}..{valid_times[-1]} "
            f"({len(valid_times)}), valid mean adv={mean_score:.4f}"
        )
        folds.append((train_lf, valid_lf))

    if not folds:
        print("  [AV] no valid folds produced; falling back to plain time CV")
        return time_cv_split(df, n_folds=n_folds, valid_frac=adv_val_ratio, embargo=embargo)
    return folds


def adversarial_cv_split(
    df: pl.LazyFrame,
    raw_feature_cols: list[str],
    adv_val_ratio: float = 0.1,
    sample_rows: int = 1_000_000,
    seed: int = 42,
    n_folds: int = 5,
    data_root: str = "data",
    embargo: int = 0,
):
    """Convenience wrapper: train AV classifier, build folds, return folds.

    When you also need the per-time_id scores for reweighting, call
    ``compute_score_by_time`` + ``build_folds_from_scores`` directly so the
    classifier is trained only once.
    """
    from src.cv import time_cv_split

    score_by_time, times, adv_scores = compute_score_by_time(
        df, raw_feature_cols, data_root=data_root, sample_rows=sample_rows, seed=seed
    )
    if not len(times):
        print("  [AV] test parquets not found; falling back to plain time CV")
        return time_cv_split(df, n_folds=n_folds, valid_frac=adv_val_ratio, embargo=embargo)
    return build_folds_from_scores(
        df, times, adv_scores, adv_val_ratio=adv_val_ratio, n_folds=n_folds, embargo=embargo
    )

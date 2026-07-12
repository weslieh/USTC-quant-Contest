import numpy as np
from lightgbm import LGBMRegressor


def build_model(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
):
    """Build a LightGBM regressor for target prediction.

    All hyperparameters are exposed so train.py can sweep them from the CLI.
    ``random_state`` is exposed for multi-seed bagging ensembles.
    """

    model = LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    return model


def weighted_r2_eval(y_true, y_pred, weight):
    """Custom LightGBM eval metric matching the competition's weighted zero-mean R².

    sklearn-API signature ``(y_true, y_pred, weight)`` — LightGBM injects the
    eval set's sample_weight automatically when ``eval_sample_weight`` is
    passed to ``fit`` (via the (X, y, w) tuple in ``eval_set``). Returns
    ``(name, value, is_higher_better)`` so early stopping optimises the exact
    leaderboard metric instead of unweighted L2.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if w.size != y_true.size:
        w = np.ones_like(y_true)

    denominator = np.sum(w * y_true * y_true)
    if denominator <= 0:
        return ("weighted_r2", 0.0, True)
    numerator = np.sum(w * (y_true - y_pred) ** 2)
    score = float(1.0 - numerator / denominator)
    return ("weighted_r2", score, True)


def build_responder_model():
    """Lighter model for responder auxiliary targets."""

    return LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def select_top_responders(
    df,
    n_top=15,
    sample_frac=0.1,
):
    """
    Select top-k responders by absolute Pearson correlation with target.
    Uses a sample to keep it fast.
    """
    responder_cols = [
        c for c in df.columns if c.startswith("responder_")
    ]
    if not responder_cols:
        return []

    n_sample = max(10000, int(len(df) * sample_frac))
    sample = df.sample(n=n_sample, seed=42) if len(df) > n_sample else df

    target = sample["target"].to_numpy().astype(np.float64)
    corrs = {}

    for c in responder_cols:
        vals = sample[c].to_numpy().astype(np.float64)
        mask = np.isfinite(vals) & np.isfinite(target)
        if mask.sum() < 100:
            continue
        corr = np.abs(np.corrcoef(vals[mask], target[mask])[0, 1])
        if np.isfinite(corr):
            corrs[c] = corr

    top = sorted(corrs, key=corrs.get, reverse=True)[:n_top]
    return top


def train_responder_models(
    X,
    df,
    responder_cols,
):
    """Train one lightweight LightGBM per responder column."""

    models = {}
    for col in responder_cols:
        y = df[col].to_numpy().astype(np.float64)
        mask = np.isfinite(y)
        if mask.sum() < 10:
            continue
        m = build_responder_model()
        m.fit(X[mask], y[mask])
        models[col] = m

    return models


def predict_responders(models, X):
    """Stack responder predictions into a feature matrix."""

    preds = []
    col_names = []
    for col, m in models.items():
        p = m.predict(X).astype(np.float64)
        preds.append(p)
        col_names.append(f"pred_{col}")

    return np.column_stack(preds) if preds else np.empty((X.shape[0], 0)), col_names

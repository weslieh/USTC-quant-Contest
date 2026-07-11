import numpy as np
from lightgbm import LGBMRegressor


def get_lgb_params(n_estimators=2000, learning_rate=0.03):
    """Return LightGBM parameter dict for lgb.train(). Equivalent to build_model() params."""

    return {
        "objective": "regression",
        "num_leaves": 64,
        "learning_rate": learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "num_threads": -1,
        "seed": 42,
        "metric": "l2",
    }


def build_model(n_estimators=2000, learning_rate=0.03):
    """Build a LightGBM regressor with sensible defaults for target prediction."""

    model = LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    return model


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

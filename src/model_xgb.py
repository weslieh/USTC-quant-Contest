import numpy as np
from xgboost import XGBRegressor

from src.metrics import weighted_zero_mean_r2


def build_xgb_model(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=100,
):
    """Build an XGBoost regressor mirroring the LightGBM setup.

    XGBoost has no ``num_leaves``; ``max_depth`` is the capacity knob. We pair
    it with ``min_child_weight`` (sum of instance weight in a leaf) which plays
    the same anti-overfit role as LightGBM's ``min_child_samples``.

    ``eval_metric`` and ``early_stopping_rounds`` are set on the estimator
    (XGBoost 2.x removed them from ``fit``). The custom weighted-R² metric and
    ``maximize=True`` make early stopping optimise the leaderboard metric.
    """

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        eval_metric=xgb_weighted_r2,
        early_stopping_rounds=early_stopping_rounds,
        maximize=True,
    )
    return model


def xgb_weighted_r2(y_true, y_pred, sample_weight=None):
    """Custom XGBoost eval metric = competition's weighted zero-mean R².

    XGBoost 3.x sklearn feval signature: ``func(y_true, y_score, sample_weight)``
    returning a plain float. ``maximize=True`` (set on the estimator) makes
    early stopping optimise this metric.
    """
    w = sample_weight
    if w is None:
        w = np.ones_like(y_true)
    return float(weighted_zero_mean_r2(np.asarray(y_true), np.asarray(y_pred), np.asarray(w)))

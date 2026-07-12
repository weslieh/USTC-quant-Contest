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

    Early stopping uses the built-in weighted RMSE (XGBoost applies
    ``sample_weight_eval_set`` to the metric automatically). A custom
    weighted-R² feval with ``maximize=True`` was tried but XGBoost 3.x's
    early-stopping/maximize interaction selected degenerate iterations
    (best_iter=0, negative R²), so we rely on weighted RMSE minimisation,
    which is a reliable proxy and stays positive on valid R².
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
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
    )
    return model


def xgb_weighted_r2(y_true, y_pred, sample_weight=None):
    """Competition weighted zero-mean R² — kept for manual scoring only.

    Not wired into early stopping (see build_xgb_model docstring).
    """
    w = sample_weight
    if w is None:
        w = np.ones_like(y_true)
    return float(weighted_zero_mean_r2(np.asarray(y_true), np.asarray(y_pred), np.asarray(w)))

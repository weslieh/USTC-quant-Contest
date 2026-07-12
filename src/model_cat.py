from catboost import CatBoostRegressor


def build_cat_model(
    iterations=2000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=3.0,
    random_seed=42,
    early_stopping_rounds=100,
):
    """Build a CatBoost regressor as the third ensemble backend.

    CatBoost's ordered boosting + symmetric trees differ structurally from
    LightGBM/XGBoost, giving genuine diversity for ensembling. Early stopping
    uses the built-in weighted RMSE (the Pool's ``weight`` is applied to the
    metric automatically) — same rationale as the XGBoost backend: a custom
    weighted-R² feval is avoided because of early-stopping/maximize quirks,
    and weighted RMSE minimisation is a reliable proxy for weighted R².

    ``allow_writing_files=False`` keeps the training dir clean (no catboost_info/
    snapshot dirs), important inside Docker.
    """

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_seed,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
        early_stopping_rounds=early_stopping_rounds,
    )
    return model

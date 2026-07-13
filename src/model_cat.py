from catboost import CatBoostRegressor, CatBoostClassifier


def build_cat_model(
    iterations=2000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=3.0,
    random_seed=42,
    early_stopping_rounds=100,
    task="regression"
):
    """Build a CatBoost model.
    """

    if task == "regression":
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
    elif task == "classification":
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=random_seed,
            thread_count=-1,
            verbose=0,
            allow_writing_files=False,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return model

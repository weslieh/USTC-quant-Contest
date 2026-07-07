from lightgbm import LGBMRegressor


def build_model():

    model = LGBMRegressor(

        objective="regression",

        n_estimators=5000,

        learning_rate=0.05,

        num_leaves=64,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        n_jobs=-1,

        verbosity=-1,

    )

    return model
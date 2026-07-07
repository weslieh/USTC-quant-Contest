import polars as pl
import lightgbm as lgb

from src.dataset import load_train
from src.cv import time_split
from src.features import get_feature_columns
from src.features import build_features
from src.model import build_model
from src.metrics import weighted_zero_mean_r2

VALID_FRACTION = 0.8

frame = load_train("data/train")


max_time = (
    frame
    .select(pl.col("time_id").max())
    .collect()
    .item()
)


times = (
    frame
    .select("time_id")
    .unique()
    .sort("time_id")
    .collect()["time_id"]
)
split_idx = int(len(times) * VALID_FRACTION)
split_time = times[split_idx]

train, valid = time_split(frame, split_time)

train_df = train.collect()
valid_df = valid.collect()

feature_cols = get_feature_columns(train)

X_train = (
    train_df
    .select(feature_cols)
    .to_numpy()
)

X_valid = (
    valid_df
    .select(feature_cols)
    .to_numpy()
)

y_train = train_df["target"].to_numpy()

y_valid = valid_df["target"].to_numpy()

w_train = train_df["weight"].to_numpy()

w_valid = valid_df["weight"].to_numpy()

train_set = lgb.Dataset(
    X_train,
    label=y_train,
    weight=w_train,
)

valid_set = lgb.Dataset(
    X_valid,
    label=y_valid,
    weight=w_valid,
)



model = build_model()

model.fit(
    X_train,
    y_train,

    sample_weight=w_train,

    eval_set=[
        (X_valid, y_valid)
    ],

    eval_metric="l2",

    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(50),
    ]
)

pred = model.predict(X_valid)

score = weighted_zero_mean_r2(
    y_valid,
    pred,
    w_valid,
)

print(score)
import polars as pl
import lightgbm as lgb
import numpy as np

from src.dataset import load_train
from src.cv import time_cv_split
from src.features import (
    get_feature_columns,
    build_rolling_features,
    fill_infinite,
)
from src.model import (
    build_model,
    select_top_responders,
    train_responder_models,
    predict_responders,
)
from src.metrics import weighted_zero_mean_r2

N_FOLDS = 3
VALID_FRAC = 0.1
N_TOP_RESPONDERS = 15
ROLLING_WINDOWS = (10, 20)

print("Loading data ...")
frame = load_train("data/train")

time_ids = (
    frame
    .select("time_id")
    .unique()
    .sort("time_id")
    .collect()["time_id"]
)

print(f"  time_ids: {len(time_ids)}")
print(f"  folds: {N_FOLDS}  valid_frac: {VALID_FRAC}")
print(f"  rolling windows: {ROLLING_WINDOWS}")
print(f"  top responders: {N_TOP_RESPONDERS}")

sample_df = load_train("data/train").head(200000).collect()
top_responders = select_top_responders(
    sample_df, n_top=N_TOP_RESPONDERS, sample_frac=0.5
)
print(f"\n  selected responders: {top_responders}")

feature_cols = get_feature_columns(frame)
print(f"  feature columns: {len(feature_cols)}")

folds = time_cv_split(frame, n_folds=N_FOLDS, valid_frac=VALID_FRAC)
print(f"\n  running {len(folds)}-fold expanding-window CV ...\n")

scores_single = []
scores_two = []

for fold_idx, (train_lf, valid_lf) in enumerate(folds):
    print(f"--- Fold {fold_idx + 1} / {len(folds)} ---")

    train_df = train_lf.collect()
    valid_df = valid_lf.collect()

    train_times = sorted(train_df["time_id"].unique().to_list())
    valid_times = sorted(valid_df["time_id"].unique().to_list())
    print(f"  train time_ids: {len(train_times)}  valid time_ids: {len(valid_times)}")
    print(f"  train rows: {len(train_df)}  valid rows: {len(valid_df)}")

    combined = pl.concat([train_df, valid_df])
    combined = build_rolling_features(combined, feature_cols, windows=ROLLING_WINDOWS)
    combined = fill_infinite(combined, feature_cols)

    train_rolled = combined.filter(pl.col("time_id").is_in(train_times))
    valid_rolled = combined.filter(pl.col("time_id").is_in(valid_times))

    all_feature_cols = [c for c in train_rolled.columns if c.startswith("feature_")]

    X_train = train_rolled.select(all_feature_cols).to_numpy().astype(np.float64)
    X_valid = valid_rolled.select(all_feature_cols).to_numpy().astype(np.float64)
    y_train = train_df["target"].to_numpy().astype(np.float64)
    y_valid = valid_df["target"].to_numpy().astype(np.float64)
    w_train = train_df["weight"].to_numpy().astype(np.float64)
    w_valid = valid_df["weight"].to_numpy().astype(np.float64)

    model_single = build_model()
    model_single.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )
    pred_single = model_single.predict(X_valid)
    score_single = weighted_zero_mean_r2(y_valid, pred_single, w_valid)
    print(f"  single-stage CV: {score_single:.6f}")
    scores_single.append(score_single)

    responder_df = train_lf.collect()
    resp_models = train_responder_models(X_train, responder_df, top_responders)
    print(f"  trained {len(resp_models)} responder models")

    if resp_models:
        resp_train, _ = predict_responders(resp_models, X_train)
        resp_valid, _ = predict_responders(resp_models, X_valid)

        if resp_train.size:
            X_train_aug = np.column_stack([X_train, resp_train])
            X_valid_aug = np.column_stack([X_valid, resp_valid])

            model_two = build_model()
            model_two.fit(
                X_train_aug, y_train,
                sample_weight=w_train,
                eval_set=[(X_valid_aug, y_valid)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
            )
            pred_two = model_two.predict(X_valid_aug)
            score_two = weighted_zero_mean_r2(y_valid, pred_two, w_valid)
            print(f"  two-stage  CV: {score_two:.6f}")
            scores_two.append(score_two)

print(f"\n{'=' * 50}")
print(f"Single-stage CV scores: {[f'{s:.6f}' for s in scores_single]}")
if scores_single:
    mean_s = sum(scores_single) / len(scores_single)
    std_s = (sum((s - mean_s) ** 2 for s in scores_single) / len(scores_single)) ** 0.5
    print(f"Single-stage Mean CV: {mean_s:.6f}  Std: {std_s:.6f}")

if scores_two:
    mean_t = sum(scores_two) / len(scores_two)
    std_t = (sum((s - mean_t) ** 2 for s in scores_two) / len(scores_two)) ** 0.5
    print(f"Two-stage   Mean CV: {mean_t:.6f}  Std: {std_t:.6f}")
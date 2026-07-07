import polars as pl
import lightgbm as lgb
import numpy as np

from src.dataset import load_train
from src.cv import time_cv_split
from src.features import (
    get_feature_columns,
    get_responder_columns,
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

N_FOLDS = 5
VALID_FRAC = 0.1
N_TOP_RESPONDERS = 15
ROLLING_WINDOWS = (10, 20)
RESPONDER_TRAIN_SPLIT = 0.7

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

# Select top responders on a sample
sample_df = load_train("data/train").head(200000).collect()
top_responders = select_top_responders(
    sample_df, n_top=N_TOP_RESPONDERS, sample_frac=0.5
)
print(f"\n  selected responders: {top_responders}")

feature_cols = get_feature_columns(frame)
print(f"  feature columns: {len(feature_cols)}")

# --------------------------------------------------
# CV loop
# --------------------------------------------------
folds = time_cv_split(
    frame,
    n_folds=N_FOLDS,
    valid_frac=VALID_FRAC,
)

print(f"\n  running {len(folds)}-fold expanding-window CV ...\n")

scores = []

for fold_idx, (train_lf, valid_lf) in enumerate(folds):
    print(f"--- Fold {fold_idx + 1} / {len(folds)} ---")

    train_df = train_lf.collect()
    valid_df = valid_lf.collect()

    train_times = sorted(train_df["time_id"].unique().to_list())
    valid_times = sorted(valid_df["time_id"].unique().to_list())
    print(f"  train time_ids: {len(train_times)}  valid time_ids: {len(valid_times)}")
    print(f"  train rows: {len(train_df)}  valid rows: {len(valid_df)}")

    # Build rolling features on combined train+valid
    # (minimal future leakage since rolling windows are small)
    combined = pl.concat([train_df, valid_df])
    combined = build_rolling_features(combined, feature_cols, windows=ROLLING_WINDOWS)
    combined = fill_infinite(combined, feature_cols)

    train_rolled = combined.filter(pl.col("time_id").is_in(train_times))
    valid_rolled = combined.filter(pl.col("time_id").is_in(valid_times))

    all_feature_cols = [c for c in train_rolled.columns if c.startswith("feature_")]

    # ---------- Stage 1: train responder models ----------
    n_early = int(len(train_times) * RESPONDER_TRAIN_SPLIT)
    early_times = train_times[:n_early]
    late_times = train_times[n_early:]

    if len(early_times) > 0 and len(late_times) > 0:
        early_df = train_rolled.filter(pl.col("time_id").is_in(early_times))
        late_df = train_rolled.filter(pl.col("time_id").is_in(late_times))

        X_early = early_df.select(all_feature_cols).to_numpy()
        X_late = late_df.select(all_feature_cols).to_numpy()
        X_valid = valid_rolled.select(all_feature_cols).to_numpy()

        print(f"  stage-1 train (early): {len(early_df)} rows")
        print(f"  stage-2 train (late): {len(late_df)} rows")

        # Use training data for responder labels (early partition only)
        responder_df = train_lf.collect()
        early_responder_df = responder_df.filter(
            pl.col("time_id").is_in(early_times)
        )

        responder_models = train_responder_models(
            X_early, early_responder_df, top_responders,
        )
        print(f"  trained {len(responder_models)} responder models")

        resp_preds_late, _ = predict_responders(responder_models, X_late)
        resp_preds_valid, _ = predict_responders(responder_models, X_valid)

        # ---------- Stage 2: train target model ----------
        if resp_preds_late.size:
            X_late_aug = np.column_stack([X_late, resp_preds_late])
            X_valid_aug = np.column_stack([X_valid, resp_preds_valid])
        else:
            X_late_aug = X_late
            X_valid_aug = X_valid

        late_df_with_y = train_lf.collect().filter(
            pl.col("time_id").is_in(late_times)
        )
        y_late = late_df_with_y["target"].to_numpy()
        w_late = late_df_with_y["weight"].to_numpy()
        y_valid_np = valid_df["target"].to_numpy()
        w_valid_np = valid_df["weight"].to_numpy()

        target_model = build_model()
        target_model.fit(
            X_late_aug, y_late,
            sample_weight=w_late,
            eval_set=[(X_valid_aug, y_valid_np)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(50),
            ],
        )

        pred = target_model.predict(X_valid_aug)
        score = weighted_zero_mean_r2(y_valid_np, pred, w_valid_np)
        print(f"  CV score: {score:.6f}")
        scores.append(score)
    else:
        print("  (single-stage fallback: not enough time_ids)")
        X_train = train_rolled.select(all_feature_cols).to_numpy()
        X_val = valid_rolled.select(all_feature_cols).to_numpy()
        y_train_np = train_df["target"].to_numpy()
        w_train_np = train_df["weight"].to_numpy()
        y_val_np = valid_df["target"].to_numpy()
        w_val_np = valid_df["weight"].to_numpy()

        model = build_model()
        model.fit(
            X_train, y_train_np,
            sample_weight=w_train_np,
            eval_set=[(X_val, y_val_np)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
        )
        pred = model.predict(X_val)
        score = weighted_zero_mean_r2(y_val_np, pred, w_val_np)
        print(f"  CV score (single-stage): {score:.6f}")
        scores.append(score)

if scores:
    mean_score = sum(scores) / len(scores)
    var_score = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    print(f"\n{'=' * 50}")
    print(f"CV scores: {[f'{s:.6f}' for s in scores]}")
    print(f"Mean CV: {mean_score:.6f}")
    print(f"Std CV:  {var_score ** 0.5:.6f}")
else:
    print("\n  No CV scores computed.")

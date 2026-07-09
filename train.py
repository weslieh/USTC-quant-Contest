import argparse
import json
from pathlib import Path

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
from src.model import build_model
from src.metrics import weighted_zero_mean_r2


MODEL_DIR = Path("strategy")
MODEL_PATH = MODEL_DIR / "model.txt"
META_PATH = MODEL_DIR / "model_meta.json"


def parse_args():
    p = argparse.ArgumentParser(description="Train LightGBM target model with time-series CV.")
    p.add_argument("--data-root", default="data", help="Data root containing manifest.json (or the train dir).")
    p.add_argument("--partitions", type=int, default=None, help="Limit to first N train partitions (memory control).")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--rolling-windows", type=int, nargs="*", default=[], help="Rolling/lag feature windows; empty = raw features only (memory-safe).")
    p.add_argument("--save-model", action="store_true", help="Save the final model + feature columns for inference.")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading data ...")
    frame = load_train(args.data_root, partitions=args.partitions)

    time_ids = (
        frame
        .select("time_id")
        .unique()
        .sort("time_id")
        .collect()["time_id"]
    )
    print(f"  time_ids: {len(time_ids)}")
    print(f"  folds: {args.n_folds}  valid_frac: {args.valid_frac}")
    print(f"  rolling windows: {args.rolling_windows}")

    feature_cols = get_feature_columns(frame)
    print(f"  feature columns: {len(feature_cols)}")

    folds = time_cv_split(frame, n_folds=args.n_folds, valid_frac=args.valid_frac)
    print(f"\n  running {len(folds)}-fold expanding-window CV ...\n")

    scores = []
    last_booster = None

    use_rolling = bool(args.rolling_windows)

    for fold_idx, (train_lf, valid_lf) in enumerate(folds):
        print(f"--- Fold {fold_idx + 1} / {len(folds)} ---")

        train_df = train_lf.collect()
        valid_df = valid_lf.collect()

        train_times = sorted(train_df["time_id"].unique().to_list())
        valid_times = sorted(valid_df["time_id"].unique().to_list())
        print(f"  train time_ids: {len(train_times)}  valid time_ids: {len(valid_times)}")
        print(f"  train rows: {len(train_df)}  valid rows: {len(valid_df)}")

        if use_rolling:
            combined = pl.concat([train_df, valid_df])
            combined = build_rolling_features(combined, feature_cols, windows=tuple(args.rolling_windows))
            combined = fill_infinite(combined, feature_cols)
            train_rolled = combined.filter(pl.col("time_id").is_in(train_times))
            valid_rolled = combined.filter(pl.col("time_id").is_in(valid_times))
            all_feature_cols = [c for c in train_rolled.columns if c.startswith("feature_")]
            X_train = train_rolled.select(all_feature_cols).to_numpy().astype(np.float32)
            X_valid = valid_rolled.select(all_feature_cols).to_numpy().astype(np.float32)
            del combined, train_rolled, valid_rolled
        else:
            all_feature_cols = list(feature_cols)
            X_train = train_df.select(all_feature_cols).to_numpy().astype(np.float32)
            X_valid = valid_df.select(all_feature_cols).to_numpy().astype(np.float32)

        y_train = train_df["target"].to_numpy().astype(np.float32)
        y_valid = valid_df["target"].to_numpy().astype(np.float32)
        w_train = train_df["weight"].to_numpy().astype(np.float32)
        w_valid = valid_df["weight"].to_numpy().astype(np.float32)

        model = build_model()
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
        )
        del X_train
        pred = model.predict(X_valid)
        del X_valid
        score = weighted_zero_mean_r2(y_valid, pred, w_valid)
        print(f"  CV score: {score:.6f}")
        scores.append(score)
        last_booster = model.booster_

    print(f"\n{'=' * 50}")
    print(f"CV scores: {[f'{s:.6f}' for s in scores]}")
    if scores:
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        print(f"Mean CV: {mean_s:.6f}  Std: {std_s:.6f}")

    if args.save_model:
        if last_booster is None:
            print("\nNo model trained; skipping save.")
            return
        # Retrain on ALL available time_ids for the deployed model so it sees
        # the most recent market regime. NOTE: rolling features multiply the
        # feature count ~5x and blow up memory on a 32GB box at full scale;
        # for the first deployable submission we keep the model on the raw
        # features only so it fits and so inference needs no state.
        print("\nRetraining final model on full data (raw features) ...")
        full_df = frame.select(feature_cols + ["weight", "target"]).collect()
        X_full = full_df.select(feature_cols).to_numpy().astype(np.float32)
        y_full = full_df["target"].to_numpy().astype(np.float32)
        w_full = full_df["weight"].to_numpy().astype(np.float32)
        deploy_feature_cols = list(feature_cols)
        del full_df

        final_model = build_model(n_estimators=max(1, model.best_iteration_ or 2000))
        final_model.fit(X_full, y_full, sample_weight=w_full, callbacks=[lgb.log_evaluation(100)])

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        final_model.booster_.save_model(str(MODEL_PATH))
        meta = {
            "feature_columns": deploy_feature_cols,
            "n_features": len(deploy_feature_cols),
            "rolling_windows": [],
            "cv_mean": mean_s if scores else None,
            "cv_std": std_s if scores else None,
        }
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved model -> {MODEL_PATH}")
        print(f"Saved meta  -> {META_PATH}  ({len(deploy_feature_cols)} features)")


if __name__ == "__main__":
    main()

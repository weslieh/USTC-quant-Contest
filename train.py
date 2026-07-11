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
    build_cross_sectional_features,
    fill_infinite,
)
from src.model import build_model
from src.metrics import weighted_zero_mean_r2


def parse_args():
    p = argparse.ArgumentParser(description="Train LightGBM target model with time-series CV.")
    p.add_argument("--data-root", default="data", help="Data root containing manifest.json (or the train dir).")
    p.add_argument("--partitions", type=int, default=None, help="Limit to first N train partitions (memory control).")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--rolling-windows", type=int, nargs="*", default=[], help="Rolling/lag feature windows; empty = raw features only (memory-safe).")
    p.add_argument("--cross-sectional", action="store_true", default=True, help="Add per-time_id cross-sectional rank/zscore/demean (default on; --no-cross-sectional to disable).")
    p.add_argument("--no-cross-sectional", dest="cross_sectional", action="store_false", help="Disable cross-sectional features.")
    p.add_argument("--gap", type=int, default=5, help="Number of time_ids to exclude between train and valid (purge gap for autocorrelation).")
    p.add_argument("--out-dir", default="strategy", help="Directory for model + CV checkpoint artifacts.")
    p.add_argument("--save-model", action="store_true", help="Save the final model + feature columns for inference.")
    p.add_argument("--fresh", action="store_true", help="Ignore existing CV checkpoints and retrain all folds.")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    cv_dir = out_dir / "cv"
    scores_path = cv_dir / "scores.json"
    model_path = out_dir / "model.txt"
    meta_path = out_dir / "model_meta.json"

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
    print(f"  folds: {args.n_folds}  valid_frac: {args.valid_frac}  gap: {args.gap}")
    print(f"  rolling windows: {args.rolling_windows}")
    print(f"  cross-sectional: {args.cross_sectional}")
    print(f"  out_dir: {out_dir}")

    feature_cols = get_feature_columns(frame)
    print(f"  feature columns: {len(feature_cols)}")

    folds = time_cv_split(frame, n_folds=args.n_folds, valid_frac=args.valid_frac, gap=args.gap)
    print(f"\n  running {len(folds)}-fold expanding-window CV ...\n")

    # Resume support: load previously completed fold scores so a spot/抢占式
    # interruption only costs the in-flight fold. CV split is deterministic
    # (sorted time_id), so per-fold indices line up across runs.
    scores = []
    if (not args.fresh) and scores_path.exists():
        try:
            saved = json.loads(scores_path.read_text(encoding="utf-8"))
            scores = list(saved.get("scores", []))
            print(f"  resuming: {len(scores)} fold(s) already completed")
        except Exception as exc:
            print(f"  could not parse {scores_path}: {exc}; starting fresh")
            scores = []

    last_booster = None
    last_best_iter = None
    use_rolling = bool(args.rolling_windows)
    use_cs = bool(args.cross_sectional)
    cv_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_lf, valid_lf) in enumerate(folds):
        if fold_idx < len(scores):
            # Already done — load its booster as the running last_booster so
            # --save-model still has something if all folds were cached.
            fold_model = cv_dir / f"fold_{fold_idx}.txt"
            if fold_model.exists():
                last_booster = lgb.Booster(model_file=str(fold_model))
            print(f"--- Fold {fold_idx + 1} / {len(folds)} (cached, score={scores[fold_idx]:.6f}) ---")
            continue

        print(f"--- Fold {fold_idx + 1} / {len(folds)} ---")

        train_df = train_lf.collect()
        valid_df = valid_lf.collect()

        train_times = sorted(train_df["time_id"].unique().to_list())
        valid_times = sorted(valid_df["time_id"].unique().to_list())
        print(f"  train time_ids: {len(train_times)}  valid time_ids: {len(valid_times)}")
        print(f"  train rows: {len(train_df)}  valid rows: {len(valid_df)}")

        # Cross-sectional features are computed per time_id within each split.
        # Since folds partition by time_id, each time_id lives entirely in
        # train OR valid — so computing per-group is strictly causal (no valid
        # info leaks into train, no future time_id used). No concat needed.
        if use_cs:
            train_df = build_cross_sectional_features(train_df, feature_cols)
            valid_df = build_cross_sectional_features(valid_df, feature_cols)

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
            # All derived cols (csrank/csz/csdm) are named feature_*_<suffix>,
            # so startswith("feature_") covers everything.
            all_feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
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
        best_iter = model.best_iteration_
        del X_train
        pred = model.predict(X_valid)
        del X_valid
        score = weighted_zero_mean_r2(y_valid, pred, w_valid)
        print(f"  CV score: {score:.6f}")

        # Checkpoint this fold before moving on.
        model.booster_.save_model(str(cv_dir / f"fold_{fold_idx}.txt"))
        scores.append(float(score))
        scores_path.write_text(json.dumps({"scores": scores}, indent=2), encoding="utf-8")
        last_booster = model.booster_
        last_best_iter = best_iter

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
        # Deploy the LAST fold's booster directly. The last fold is trained on
        # the most recent (largest) expanding window, so it already sees the
        # latest market regime — a separate full-data retrain adds no signal
        # and a full LazyFrame collect here has been observed to deadlock
        # (0% CPU, no I/O) after the CV folds hold the data. Skipping it is
        # both faster and safer. last_booster is either the just-trained
        # last fold or the one loaded from its checkpoint on resume.
        print("\nSaving last fold's booster as the deploy model ...")

        # Determine the full feature set the booster was trained on. On a
        # fresh run all_feature_cols is set by the last fold; on a fully-cached
        # resume it is undefined, so reconstruct it from the args + schema.
        if "all_feature_cols" not in dir() or not all_feature_cols:
            cols = list(feature_cols)
            if use_cs:
                cols += [
                    s
                    for c in feature_cols
                    for s in (f"{c}_csrank", f"{c}_csz", f"{c}_csdm")
                ]
            if use_rolling:
                cols += [
                    s
                    for c in feature_cols
                    for s in (f"{c}_lag1",)
                ]
                cols += [
                    s
                    for c in feature_cols
                    for w in args.rolling_windows
                    for s in (f"{c}_rm_{w}", f"{c}_rs_{w}")
                ]
            all_feature_cols = cols
        deploy_feature_cols = list(all_feature_cols)

        out_dir.mkdir(parents=True, exist_ok=True)
        last_booster.save_model(str(model_path))
        meta = {
            "feature_columns": deploy_feature_cols,
            "n_features": len(deploy_feature_cols),
            "rolling_windows": list(args.rolling_windows) if use_rolling else [],
            "cross_sectional": use_cs,
            "gap": args.gap,
            "cv_mean": float(np.mean(scores)) if scores else None,
            "cv_std": float(np.std(scores)) if scores else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved model -> {model_path}")
        print(f"Saved meta  -> {meta_path}  ({len(deploy_feature_cols)} features)")


if __name__ == "__main__":
    main()

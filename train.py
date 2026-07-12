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
    build_cross_sectional_features,
    build_rolling_features,
    fill_infinite,
)
from src.model import build_model, weighted_r2_eval
from src.metrics import weighted_zero_mean_r2

def parse_args():
    p = argparse.ArgumentParser(description="Train LightGBM target model with time-series CV.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--partitions", type=int, default=None, help="Limit to first N train partitions.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--embargo", type=int, default=0, help="Time-id gap between train end and valid start.")
    # Feature engineering
    p.add_argument("--cs-topk", type=int, default=25, help="Top-K raw features for cross-sectional derivs (0=off).")
    p.add_argument("--rolling-windows", type=int, nargs="*", default=[], help="Rolling windows (empty=off).")
    p.add_argument("--rolling-topk", type=int, default=20, help="Top-K raw features for rolling (0=all 323).")
    p.add_argument("--importance-sample", type=int, default=1_000_000, help="Rows for pilot importance.")
    # Hyperparameters
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--n-est", type=int, default=2000)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--feature-frac", type=float, default=0.8)
    p.add_argument("--bagging-frac", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=0.1)
    p.add_argument("--early-stopping-rounds", type=int, default=100)
    # Backend
    p.add_argument("--backend", choices=["lgb", "xgb", "cat"], default="lgb",
                   help="Trainer backend: lgb=LightGBM, xgb=XGBoost, cat=CatBoost.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the model (multi-seed bagging: run with several seeds).")
    # XGBoost-specific hyperparameters (ignored when --backend lgb)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-min-child-weight", type=float, default=5.0)
    p.add_argument("--xgb-subsample", type=float, default=0.8)
    p.add_argument("--xgb-colsample", type=float, default=0.8)
    # CatBoost-specific hyperparameters (ignored when --backend is not cat)
    p.add_argument("--cat-depth", type=int, default=7)
    p.add_argument("--cat-l2-leaf-reg", type=float, default=3.0)
    # IO
    p.add_argument("--out-dir", default="strategy")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--fresh", action="store_true")
    return p.parse_args()


def select_topk_by_importance(frame, raw_feature_cols, k, sample_rows, build_kwargs):
    """Pilot-fit on a sample with raw features, return top-k feature names by gain."""
    if k <= 0 or k >= len(raw_feature_cols):
        return list(raw_feature_cols)
    print(f"  pilot importance fit on up to {sample_rows} rows ...")
    sample = frame.head(sample_rows).collect()
    X = sample.select(raw_feature_cols).to_numpy().astype(np.float32)
    y = sample["target"].to_numpy().astype(np.float32)
    w = sample["weight"].to_numpy().astype(np.float32)
    pilot = build_model(n_estimators=300, learning_rate=0.05, num_leaves=63,
                        min_child_samples=100, **{kk: vv for kk, vv in build_kwargs.items()
                                                  if kk in ("feature_fraction", "bagging_fraction",
                                                            "bagging_freq", "reg_alpha", "reg_lambda")})
    pilot.fit(X, y, sample_weight=w)
    imp = dict(zip(raw_feature_cols, pilot.feature_importances_.tolist()))
    top = sorted(imp, key=imp.get, reverse=True)[:k]
    del sample, X, y, w, pilot
    return top


def build_fold_features(train_df, valid_df, raw_cols, cs_source_cols, rolling_source_cols, rolling_windows):
    """Attach cross-sectional + rolling features to a fold's train/valid.

    Cross-sectional features are stateless per time_id. Rolling features are
    computed independently on train and valid (each fresh-starting) so the
    valid fold mimics inference, where the per-asset history buffer starts
    empty — no train history leaks into valid rolling stats.
    """
    new_cs_cols = []
    new_roll_cols = []

    def _attach(df, cs_src, roll_src, wins):
        out = df
        cs_cols = []
        if cs_src:
            out, cs_cols = build_cross_sectional_features(out, cs_src)
        # Rolling column names are deterministic from source + windows.
        roll_cols = []
        if roll_src and wins:
            out = build_rolling_features(out, roll_src, windows=tuple(wins))
            for s in roll_src:
                roll_cols.append(f"{s}_lag1")
                for w in wins:
                    roll_cols += [f"{s}_rm_{w}", f"{s}_rs_{w}"]
        return out, cs_cols, roll_cols

    train_out, train_cs, train_roll = _attach(train_df, cs_source_cols, rolling_source_cols, rolling_windows)
    valid_out, valid_cs, valid_roll = _attach(valid_df, cs_source_cols, rolling_source_cols, rolling_windows)
    # cs/roll column name sets are identical across train/valid (same sources).
    return train_out, valid_out, train_cs, train_roll


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    cv_dir = out_dir / "cv"
    scores_path = cv_dir / "scores.json"
    meta_path = out_dir / "model_meta.json"

    print("Loading data ...")
    frame = load_train(args.data_root, partitions=args.partitions)

    time_ids = (
        frame.select("time_id").unique().sort("time_id").collect()["time_id"]
    )
    print(f"  time_ids: {len(time_ids)}")
    print(f"  folds: {args.n_folds}  valid_frac: {args.valid_frac}  embargo: {args.embargo}")

    raw_feature_cols = get_feature_columns(frame)
    print(f"  raw feature columns: {len(raw_feature_cols)}")

    # Sanitize rolling windows: drop non-positive values; empty means off.
    rolling_windows = [w for w in args.rolling_windows if w > 0]
    if rolling_windows != args.rolling_windows:
        print(f"  rolling_windows sanitized -> {rolling_windows}")
    print(f"  cs_topk: {args.cs_topk}  rolling_windows: {rolling_windows}  rolling_topk: {args.rolling_topk}")

    build_kwargs = dict(
        num_leaves=args.num_leaves, learning_rate=args.lr, n_estimators=args.n_est,
        min_child_samples=args.min_child_samples, feature_fraction=args.feature_frac,
        bagging_fraction=args.bagging_frac, bagging_freq=args.bagging_freq,
        reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
        random_state=args.seed,
    )

    # Selective top-K source columns for cs / rolling (fixed across folds & inference).
    cs_source_cols = []
    rolling_source_cols = []
    need_selection = args.cs_topk > 0 or (rolling_windows and args.rolling_topk > 0)
    if need_selection:
        all_top = select_topk_by_importance(
            frame, raw_feature_cols,
            k=max(args.cs_topk, args.rolling_topk if args.rolling_topk > 0 else 0),
            sample_rows=args.importance_sample, build_kwargs=build_kwargs,
        )
        if args.cs_topk > 0:
            cs_source_cols = all_top[:args.cs_topk]
            print(f"  cs source cols ({len(cs_source_cols)}): {cs_source_cols[:5]} ...")
        if rolling_windows and args.rolling_topk > 0:
            rolling_source_cols = all_top[:args.rolling_topk]
            print(f"  rolling source cols ({len(rolling_source_cols)}): {rolling_source_cols[:5]} ...")
        elif rolling_windows:
            rolling_source_cols = list(raw_feature_cols)

    folds = time_cv_split(frame, n_folds=args.n_folds, valid_frac=args.valid_frac, embargo=args.embargo)
    print(f"\n  running {len(folds)}-fold expanding-window CV ...\n")

    scores = []
    if (not args.fresh) and scores_path.exists():
        try:
            saved = json.loads(scores_path.read_text(encoding="utf-8"))
            scores = list(saved.get("scores", []))
            print(f"  resuming: {len(scores)} fold(s) already completed")
        except Exception as exc:
            print(f"  could not parse {scores_path}: {exc}; starting fresh")
            scores = []

    cv_dir.mkdir(parents=True, exist_ok=True)
    fold_boosters = []  # (fold_idx, (backend, booster_or_path), best_iter)
    target_std = None

    for fold_idx, (train_lf, valid_lf) in enumerate(folds):
        if fold_idx < len(scores):
            if args.backend == "xgb":
                fmodel = cv_dir / f"fold_{fold_idx}.json"
                if fmodel.exists():
                    fold_boosters.append((fold_idx, ("xgb", str(fmodel)), None))
            elif args.backend == "cat":
                fmodel = cv_dir / f"fold_{fold_idx}.cbm"
                if fmodel.exists():
                    fold_boosters.append((fold_idx, ("cat", str(fmodel)), None))
            else:
                fmodel = cv_dir / f"fold_{fold_idx}.txt"
                if fmodel.exists():
                    fold_boosters.append((fold_idx, ("lgb", lgb.Booster(model_file=str(fmodel))), None))
            print(f"--- Fold {fold_idx + 1} / {len(folds)} (cached, score={scores[fold_idx]:.6f}) ---")
            continue

        print(f"--- Fold {fold_idx + 1} / {len(folds)} ---")

        train_df = train_lf.collect()
        valid_df = valid_lf.collect()

        train_times = sorted(train_df["time_id"].unique().to_list())
        valid_times = sorted(valid_df["time_id"].unique().to_list())
        print(f"  train time_ids: {len(train_times)}  valid time_ids: {len(valid_times)}")
        print(f"  train rows: {len(train_df)}  valid rows: {len(valid_df)}")

        train_df, valid_df, cs_cols, roll_cols = build_fold_features(
            train_df, valid_df, raw_feature_cols, cs_source_cols, rolling_source_cols, rolling_windows,
        )
        # fill inf -> 0 on engineered cols; raw NaN left for LGBM native handling
        if cs_cols or roll_cols:
            train_df = fill_infinite(train_df, cs_cols + roll_cols)
            valid_df = fill_infinite(valid_df, cs_cols + roll_cols)

        all_feature_cols = list(raw_feature_cols) + cs_cols + roll_cols
        X_train = train_df.select(all_feature_cols).to_numpy().astype(np.float32)
        X_valid = valid_df.select(all_feature_cols).to_numpy().astype(np.float32)
        y_train = train_df["target"].to_numpy().astype(np.float32)
        y_valid = valid_df["target"].to_numpy().astype(np.float32)
        w_train = train_df["weight"].to_numpy().astype(np.float32)
        w_valid = valid_df["weight"].to_numpy().astype(np.float32)

        if target_std is None:
            target_std = float(np.std(y_train))

        if args.backend == "xgb":
            from src.model_xgb import build_xgb_model
            model = build_xgb_model(
                n_estimators=args.n_est, learning_rate=args.lr,
                max_depth=args.xgb_max_depth, min_child_weight=args.xgb_min_child_weight,
                subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample,
                reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
                early_stopping_rounds=args.early_stopping_rounds,
                random_state=args.seed,
            )
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_valid, y_valid)],
                sample_weight_eval_set=[w_valid],
                verbose=False,
            )
            best_iter = getattr(model, "best_iteration", None)
            del X_train
            pred = model.predict(X_valid)
            del X_valid
            score = weighted_zero_mean_r2(y_valid, pred, w_valid)
            print(f"  CV score: {score:.6f}  best_iter: {best_iter}")
            # Save XGB fold as json (native format); load with XGBRegressor in inference.
            model.save_model(str(cv_dir / f"fold_{fold_idx}.json"))
            scores.append(float(score))
            scores_path.write_text(json.dumps({"scores": scores}, indent=2), encoding="utf-8")
            fold_boosters.append((fold_idx, ("xgb", str(cv_dir / f"fold_{fold_idx}.json")), best_iter))
            del train_df, valid_df, y_train, y_valid, w_train, w_valid, model
            continue

        if args.backend == "cat":
            from src.model_cat import build_cat_model
            from catboost import Pool
            model = build_cat_model(
                iterations=args.n_est, learning_rate=args.lr,
                depth=args.cat_depth, l2_leaf_reg=args.cat_l2_leaf_reg,
                random_seed=args.seed,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            train_pool = Pool(X_train, y_train, weight=w_train)
            valid_pool = Pool(X_valid, y_valid, weight=w_valid)
            model.fit(train_pool, eval_set=valid_pool)
            best_iter = model.best_iteration_
            del X_train
            pred = model.predict(X_valid)
            del X_valid
            score = weighted_zero_mean_r2(y_valid, pred, w_valid)
            print(f"  CV score: {score:.6f}  best_iter: {best_iter}")
            model.save_model(str(cv_dir / f"fold_{fold_idx}.cbm"))
            scores.append(float(score))
            scores_path.write_text(json.dumps({"scores": scores}, indent=2), encoding="utf-8")
            fold_boosters.append((fold_idx, ("cat", str(cv_dir / f"fold_{fold_idx}.cbm")), best_iter))
            del train_df, valid_df, y_train, y_valid, w_train, w_valid, model, train_pool, valid_pool
            continue

        model = build_model(**build_kwargs)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid, w_valid)],
            eval_metric=weighted_r2_eval,
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False), lgb.log_evaluation(50)],
        )
        best_iter = model.best_iteration_
        del X_train
        pred = model.predict(X_valid)
        del X_valid
        score = weighted_zero_mean_r2(y_valid, pred, w_valid)
        print(f"  CV score: {score:.6f}  best_iter: {best_iter}")

        model.booster_.save_model(str(cv_dir / f"fold_{fold_idx}.txt"))
        scores.append(float(score))
        scores_path.write_text(json.dumps({"scores": scores}, indent=2), encoding="utf-8")
        fold_boosters.append((fold_idx, ("lgb", model.booster_), best_iter))
        del train_df, valid_df, y_train, y_valid, w_train, w_valid, model

    print(f"\n{'=' * 50}")
    print(f"CV scores: {[f'{s:.6f}' for s in scores]}")
    if scores:
        print(f"Mean CV: {float(np.mean(scores)):.6f}  Std: {float(np.std(scores)):.6f}")

    if args.save_model:
        if not fold_boosters:
            print("\nNo model trained; skipping save.")
            return
        import shutil
        print(f"\nSaving {len(fold_boosters)} fold boosters ({args.backend}) as the deploy ensemble ...")
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_fold_files = []
        for fold_idx, (backend, booster_or_path), _ in fold_boosters:
            if backend == "xgb":
                fname = f"booster_fold_{fold_idx}.json"
                shutil.copyfile(booster_or_path, str(out_dir / fname))
            elif backend == "cat":
                fname = f"booster_fold_{fold_idx}.cbm"
                shutil.copyfile(booster_or_path, str(out_dir / fname))
            else:
                fname = f"booster_fold_{fold_idx}.txt"
                booster_or_path.save_model(str(out_dir / fname))
            saved_fold_files.append(fname)

        # Final feature column order = raw + cs + rolling (matches build_fold_features).
        cs_cols, roll_cols = [], []
        if cs_source_cols:
            cs_cols = []
            for s in cs_source_cols:
                cs_cols += [f"{s}_cs_rank", f"{s}_cs_z", f"{s}_cs_dm"]
        if rolling_source_cols and rolling_windows:
            roll_cols = []
            for s in rolling_source_cols:
                roll_cols.append(f"{s}_lag1")
                for w in rolling_windows:
                    roll_cols += [f"{s}_rm_{w}", f"{s}_rs_{w}"]
        all_feature_cols = list(raw_feature_cols) + cs_cols + roll_cols

        meta = {
            "backend": args.backend,
            "feature_columns": all_feature_cols,
            "raw_feature_columns": list(raw_feature_cols),
            "cs_source_columns": cs_source_cols,
            "cs_feature_columns": cs_cols,
            "rolling_source_columns": rolling_source_cols,
            "rolling_feature_columns": roll_cols,
            "rolling_windows": list(rolling_windows),
            "n_folds": len(fold_boosters),
            "fold_files": saved_fold_files,
            "target_std": target_std,
            "cv_mean": float(np.mean(scores)) if scores else None,
            "cv_std": float(np.std(scores)) if scores else None,
            "hparams": build_kwargs,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved {len(fold_boosters)} boosters -> {out_dir}")
        print(f"Saved meta -> {meta_path}  ({len(all_feature_cols)} features)")


if __name__ == "__main__":
    main()

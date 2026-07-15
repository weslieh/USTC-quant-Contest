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
from src.interactions import make_pairs, build_interaction_features, interaction_column_names
from src.target_transform import target_rank_per_time, build_inverse_cdf_lut, inverse_cdf_map
from src.model import build_model, weighted_r2_eval
from src.metrics import weighted_zero_mean_r2

def parse_args():
    p = argparse.ArgumentParser(description="Train LightGBM target model with time-series CV.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--partitions", type=int, default=None, help="Limit to first N train partitions.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--embargo", type=int, default=0, help="Time-id gap between train end and valid start.")
    p.add_argument("--adv-val-ratio", type=float, default=0.0, help="Use adversarial validation split instead of time CV. 0=off.")
    p.add_argument("--av-reweight", action="store_true", help="Reweight train samples by AV odds-ratio (test-likeness) to correct drift. Train-only; inference unchanged.")
    p.add_argument("--reweight-clip-quantile", type=float, default=0.99, help="Winsorize p_like_test at this quantile before odds-ratio (anti-explosion when AUC~1.0).")
    p.add_argument("--multi-task", action="store_true", help="Train dual models: target direction (classification) and target value (regression).")
    p.add_argument("--neutralize-alpha", type=float, default=0.0, help="Proportion of neutralization to apply during inference against drift features. 0=off.")
    p.add_argument("--neutralize-topk", type=int, default=10, help="Number of top drift features to neutralize against.")
    # Feature engineering
    p.add_argument("--cs-topk", type=int, default=25, help="Top-K raw features for cross-sectional derivs (0=off).")
    p.add_argument("--rolling-windows", type=int, nargs="*", default=[], help="Rolling windows (empty=off).")
    p.add_argument("--rolling-topk", type=int, default=20, help="Top-K raw features for rolling (0=all 323).")
    p.add_argument("--interaction-topk", type=int, default=0, help="Top-K raw features for pairwise mul/div interactions (0=off). ~K*(K-1)/2 pairs * 2 cols.")
    p.add_argument("--target-transform", choices=["none", "rank"], default="none", help="Transform the training target. 'rank' = per-time_id rank percentile; inverse-CDF LUT stored for inference.")
    p.add_argument("--asset-as-categorical", action="store_true", help="Prepend asset_id as the first feature column. LightGBM treats it as categorical (optimal per-category splits); XGB/CatBoost treat it as a numeric column. Column order is identical across backends.")
    p.add_argument("--per-asset", action="store_true", help="Train one independent model per asset_id (15 models, each 5-fold). Inference routes by asset_id. Forces raw-only features (no asset_id column / cs / rolling / interactions).")
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


def build_fold_features(train_df, valid_df, raw_cols, cs_source_cols, rolling_source_cols, rolling_windows, interaction_pairs=None):
    """Attach cross-sectional + rolling + interaction features to a fold's train/valid.

    Cross-sectional features are stateless per time_id. Rolling features are
    computed independently on train and valid (each fresh-starting) so the
    valid fold mimics inference, where the per-asset history buffer starts
    empty — no train history leaks into valid rolling stats. Interaction
    features are within-row (feature axis), so they are identical on both.
    """
    new_cs_cols = []
    new_roll_cols = []
    new_inter_cols = []

    def _attach(df, cs_src, roll_src, wins, inter_pairs):
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
        inter_cols = []
        if inter_pairs:
            out, inter_cols = build_interaction_features(out, inter_pairs)
        return out, cs_cols, roll_cols, inter_cols

    train_out, train_cs, train_roll, train_inter = _attach(
        train_df, cs_source_cols, rolling_source_cols, rolling_windows, interaction_pairs)
    valid_out, valid_cs, valid_roll, valid_inter = _attach(
        valid_df, cs_source_cols, rolling_source_cols, rolling_windows, interaction_pairs)
    # cs/roll/inter column name sets are identical across train/valid (same sources).
    return train_out, valid_out, train_cs, train_roll, train_inter


def train_per_asset(args):
    """Train one independent model per asset_id (LGB-only for now).

    Each of the 15 assets gets its own 5-fold expanding-window time CV with
    the same embargo/early-stopping as the global model. Boosters are saved
    per-asset under ``<out_dir>/asset_<id>/``; a top-level model_meta.json
    records the shared feature spec and the asset directory list so the
    inference side can route rows by asset_id.

    Forces raw-only features: per-asset models already encode asset identity
    by construction, so asset_id column / cs / rolling / interactions are off.
    """
    from src.cv import time_cv_split
    from src.model import build_model, weighted_r2_eval
    from src.metrics import weighted_zero_mean_r2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "model_meta.json"

    print("Loading data ...")
    frame = load_train(args.data_root, partitions=args.partitions)
    raw_feature_cols = get_feature_columns(frame)
    print(f"  raw feature columns: {len(raw_feature_cols)}")

    build_kwargs = dict(
        num_leaves=args.num_leaves, learning_rate=args.lr, n_estimators=args.n_est,
        min_child_samples=args.min_child_samples, feature_fraction=args.feature_frac,
        bagging_fraction=args.bagging_frac, bagging_freq=args.bagging_freq,
        reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
        random_state=args.seed,
    )

    asset_ids = sorted(frame.select("asset_id").unique().collect()["asset_id"].to_list())
    print(f"  assets: {asset_ids}  ({len(asset_ids)} models)")

    target_std = None
    asset_dirs = []
    per_asset_scores = {}
    all_valid_preds = []  # (rows, pred, y, w) for overall weighted score

    for aid in asset_ids:
        print(f"\n{'=' * 50}\n=== asset_id {aid} ===")
        asset_frame = frame.filter(pl.col("asset_id") == aid)
        folds = time_cv_split(
            asset_frame, n_folds=args.n_folds, valid_frac=args.valid_frac, embargo=args.embargo,
        )
        asset_out = out_dir / f"asset_{aid}"
        asset_cv = asset_out / "cv"
        asset_cv.mkdir(parents=True, exist_ok=True)
        asset_scores = []
        asset_boosters = []

        for fold_idx, (train_lf, valid_lf) in enumerate(folds):
            print(f"  --- Fold {fold_idx + 1} / {len(folds)} ---")
            train_df = train_lf.collect()
            valid_df = valid_lf.collect()
            X_train = train_df.select(raw_feature_cols).to_numpy().astype(np.float32)
            X_valid = valid_df.select(raw_feature_cols).to_numpy().astype(np.float32)
            y_train = train_df["target"].to_numpy().astype(np.float32)
            y_valid = valid_df["target"].to_numpy().astype(np.float32)
            w_train = train_df["weight"].to_numpy().astype(np.float32)
            w_valid = valid_df["weight"].to_numpy().astype(np.float32)

            if target_std is None:
                target_std = float(np.std(y_train))

            model = build_model(**build_kwargs)
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_valid, y_valid, w_valid)],
                eval_metric=weighted_r2_eval,
                callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False), lgb.log_evaluation(100)],
            )
            best_iter = model.best_iteration_
            del X_train
            pred = model.predict(X_valid)
            score = weighted_zero_mean_r2(y_valid, pred, w_valid)
            print(f"  CV score: {score:.6f}  best_iter: {best_iter}")
            model.booster_.save_model(str(asset_cv / f"fold_{fold_idx}.txt"))
            asset_scores.append(float(score))
            asset_boosters.append((fold_idx, model.booster_))
            all_valid_preds.append((pred, y_valid, w_valid))
            del train_df, valid_df, y_train, X_valid, model

        mean_cv = float(np.mean(asset_scores)) if asset_scores else 0.0
        per_asset_scores[int(aid)] = asset_scores
        print(f"  asset {aid} mean CV: {mean_cv:.6f}")

        # Save this asset's fold boosters.
        for fold_idx, booster in asset_boosters:
            booster.save_model(str(asset_out / f"booster_fold_{fold_idx}.txt"))
        asset_dirs.append({"asset_id": int(aid), "dir": f"asset_{aid}",
                           "n_folds": len(asset_boosters), "cv_mean": mean_cv})

    # Overall weighted R² across all assets' OOF valid predictions.
    overall = 0.0
    if all_valid_preds:
        all_pred = np.concatenate([p for p, _, _ in all_valid_preds])
        all_y = np.concatenate([y for _, y, _ in all_valid_preds])
        all_w = np.concatenate([w for _, _, w in all_valid_preds])
        overall = float(weighted_zero_mean_r2(all_y, all_pred, all_w))
    print(f"\n{'=' * 50}")
    print(f"Per-asset CV means: {[round(float(np.mean(per_asset_scores[a])), 6) for a in sorted(per_asset_scores)]}")
    print(f"Overall OOF weighted R2: {overall:.6f}")

    meta = {
        "backend": "lgb_perasset",
        "per_asset": True,
        "feature_columns": list(raw_feature_cols),
        "raw_feature_columns": list(raw_feature_cols),
        "n_assets": len(asset_ids),
        "asset_dirs": asset_dirs,
        "n_folds": args.n_folds,
        "target_std": target_std,
        "cv_mean": overall,
        "per_asset_cv": {str(a): s for a, s in per_asset_scores.items()},
        "hparams": build_kwargs,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved per-asset meta -> {meta_path}")


def main():
    args = parse_args()

    if getattr(args, "per_asset", False):
        return train_per_asset(args)

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

    drift_features = []
    if args.neutralize_alpha > 0 and args.neutralize_topk > 0:
        from src.drift import compute_drift_rank
        from src.dataset import _partition_paths
        train_paths = _partition_paths(args.data_root, "train")
        test_paths = _partition_paths(args.data_root, "test")
        drift_rank = compute_drift_rank(train_paths, test_paths, list(raw_feature_cols), seed=args.seed)
        drift_features = drift_rank[:args.neutralize_topk]
        print(f"  neutralization drift features ({len(drift_features)}): {drift_features}")

    build_kwargs = dict(
        num_leaves=args.num_leaves, learning_rate=args.lr, n_estimators=args.n_est,
        min_child_samples=args.min_child_samples, feature_fraction=args.feature_frac,
        bagging_fraction=args.bagging_frac, bagging_freq=args.bagging_freq,
        reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
        random_state=args.seed,
    )

    # Selective top-K source columns for cs / rolling / interactions (fixed across folds & inference).
    cs_source_cols = []
    rolling_source_cols = []
    interaction_pairs = []
    need_selection = (
        args.cs_topk > 0
        or (rolling_windows and args.rolling_topk > 0)
        or args.interaction_topk > 0
    )
    if need_selection:
        all_top = select_topk_by_importance(
            frame, raw_feature_cols,
            k=max(args.cs_topk, args.rolling_topk if args.rolling_topk > 0 else 0,
                  args.interaction_topk),
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
        if args.interaction_topk > 0:
            inter_src = all_top[:args.interaction_topk]
            interaction_pairs = make_pairs(inter_src)
            n_inter_cols = 2 * len(interaction_pairs)
            print(
                f"  interaction source cols ({len(inter_src)}): {inter_src[:5]} ... "
                f"-> {len(interaction_pairs)} pairs, {n_inter_cols} new cols"
            )

    # Adversarial validation: either select test-like validation rows
    # (--adv-val-ratio), or reweight train samples by test-likeness
    # (--av-reweight), or both. Train the AV classifier once and reuse the
    # per-time_id scores for both purposes.
    use_av = args.adv_val_ratio > 0 or args.av_reweight
    score_by_time = {}
    if use_av:
        from src.adversarial_cv import compute_score_by_time, build_folds_from_scores
        print(f"\n  computing adversarial test-likeness (real test features) ...\n")
        score_by_time, av_times, av_scores = compute_score_by_time(
            frame, list(raw_feature_cols),
            data_root=args.data_root,
            sample_rows=args.importance_sample,
            seed=args.seed,
        )

    if args.adv_val_ratio > 0:
        print(f"  building AV validation folds (ratio={args.adv_val_ratio}) ...\n")
        folds = build_folds_from_scores(
            frame, av_times, av_scores,
            adv_val_ratio=args.adv_val_ratio,
            n_folds=args.n_folds,
            embargo=args.embargo,
        )
    elif args.av_reweight:
        # Reweight only — keep the plain time CV folds.
        folds = time_cv_split(frame, n_folds=args.n_folds, valid_frac=args.valid_frac, embargo=args.embargo)
        print(f"\n  running {len(folds)}-fold expanding-window CV (AV reweight on) ...\n")
    else:
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
    target_lut = None  # inverse-CDF LUT for target-transform=rank (built fold 0)

    def _score_valid(pred, y_valid_raw_, w_valid_):
        """Compute weighted R² in the ORIGINAL target scale.

        Under target-transform=rank the model predicts a rank in [0,1]; we
        map it back via the LUT before scoring so CV tracks the public
        leaderboard. For none/multi-task, pred is already in target scale.
        """
        if args.target_transform == "rank" and target_lut is not None:
            pred = inverse_cdf_map(pred, target_lut)
        return weighted_zero_mean_r2(y_valid_raw_, pred, w_valid_)

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

        train_df, valid_df, cs_cols, roll_cols, inter_cols = build_fold_features(
            train_df, valid_df, raw_feature_cols, cs_source_cols, rolling_source_cols, rolling_windows,
            interaction_pairs=interaction_pairs,
        )
        # fill inf -> 0 on engineered cols; raw NaN left for LGBM native handling
        engineered_cols = cs_cols + roll_cols + inter_cols
        if engineered_cols:
            train_df = fill_infinite(train_df, engineered_cols)
            valid_df = fill_infinite(valid_df, engineered_cols)

        all_feature_cols = list(raw_feature_cols) + cs_cols + roll_cols + inter_cols
        # Optionally prepend asset_id as a feature column (index 0). LightGBM
        # uses it as categorical; XGB/Cat treat it as numeric. Column order is
        # identical across backends so the inference side can reproduce it.
        asset_as_cat = bool(args.asset_as_categorical)
        if asset_as_cat:
            all_feature_cols = ["asset_id"] + all_feature_cols
        X_train = train_df.select(all_feature_cols).to_numpy().astype(np.float32)
        X_valid = valid_df.select(all_feature_cols).to_numpy().astype(np.float32)
        y_train_raw = train_df["target"].to_numpy().astype(np.float32)
        y_valid_raw = valid_df["target"].to_numpy().astype(np.float32)
        y_train = y_train_raw
        y_valid = y_valid_raw
        w_train = train_df["weight"].to_numpy().astype(np.float32)
        w_valid = valid_df["weight"].to_numpy().astype(np.float32)

        # AV covariate-shift reweighting: multiply train weights by the
        # odds-ratio of test-likeness (train-only; valid keeps original weights
        # so CV still reports the public-leaderboard metric).
        if args.av_reweight and score_by_time:
            from src.reweight import time_id_to_weights
            train_time_ids = train_df["time_id"].to_numpy()
            rw = time_id_to_weights(
                train_time_ids, score_by_time,
                clip_quantile=args.reweight_clip_quantile,
            )
            w_train = w_train * rw
            print(f"  [reweight] w_train mean={w_train.mean():.4f} "
                  f"(odds-ratio clip q={args.reweight_clip_quantile})")

        # target_std and the inverse-CDF LUT are in the ORIGINAL target scale
        # (the clip bound and the inverse map must both be original-scale).
        if target_std is None:
            target_std = float(np.std(y_train_raw))
            if args.target_transform == "rank":
                target_lut = build_inverse_cdf_lut(y_train_raw, n_points=1001)
                print(f"  [target-transform] rank mode; built inverse-CDF LUT "
                      f"(target_std={target_std:.4f})")

        # Transform the regression target to per-time_id rank percentile so the
        # model learns a scale-invariant ordering. Inference maps the predicted
        # rank back via the LUT; CV scores in original space (see _score_valid).
        if args.target_transform == "rank":
            y_train = target_rank_per_time(
                train_df["time_id"].to_numpy(), y_train_raw)
            y_valid = target_rank_per_time(
                valid_df["time_id"].to_numpy(), y_valid_raw)

        if args.backend == "xgb":
            from src.model_xgb import build_xgb_model

            if getattr(args, "multi_task", False):
                # Target binary: 1 if target > 0 else 0
                y_train_bin = (y_train > 0).astype(np.float32)
                y_valid_bin = (y_valid > 0).astype(np.float32)

                print("  Training Classifier...")
                clf = build_xgb_model(
                    n_estimators=args.n_est, learning_rate=args.lr,
                    max_depth=args.xgb_max_depth, min_child_weight=args.xgb_min_child_weight,
                    subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample,
                    reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
                    early_stopping_rounds=args.early_stopping_rounds,
                    random_state=args.seed,
                    task="classification"
                )
                clf.fit(
                    X_train, y_train_bin,
                    sample_weight=w_train,
                    eval_set=[(X_valid, y_valid_bin)],
                    sample_weight_eval_set=[w_valid],
                    verbose=False,
                )

                print("  Training Regressor...")
                reg = build_xgb_model(
                    n_estimators=args.n_est, learning_rate=args.lr,
                    max_depth=args.xgb_max_depth, min_child_weight=args.xgb_min_child_weight,
                    subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample,
                    reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda,
                    early_stopping_rounds=args.early_stopping_rounds,
                    random_state=args.seed,
                    task="regression"
                )
                reg.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_valid, y_valid)],
                    sample_weight_eval_set=[w_valid],
                    verbose=False,
                )

                best_iter = getattr(reg, "best_iteration", None)
                del X_train

                pred_prob = clf.predict_proba(X_valid)[:, 1]
                pred_val = reg.predict(X_valid)
                # Combine predictions: simple scale or probability weighting
                # Using a heuristic: probability of positive * raw prediction magnitude
                # A more refined approach tunes this mapping. For now we use the probability.
                pred = pred_prob * pred_val
                del X_valid

                score = _score_valid(pred, y_valid_raw, w_valid)
                print(f"  CV score: {score:.6f}  best_iter(reg): {best_iter}")

                clf.save_model(str(cv_dir / f"fold_{fold_idx}_clf.json"))
                reg.save_model(str(cv_dir / f"fold_{fold_idx}_reg.json"))
                scores.append(float(score))
                scores_path.write_text(json.dumps({"scores": scores}, indent=2), encoding="utf-8")
                fold_boosters.append((fold_idx, ("xgb_mt", str(cv_dir / f"fold_{fold_idx}")), best_iter))
                del train_df, valid_df, y_train, y_valid, w_train, w_valid, clf, reg
                continue

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

            if getattr(args, "multi_task", False):
                y_train_bin = (y_train > 0).astype(np.float32)
                y_valid_bin = (y_valid > 0).astype(np.float32)

                print("  Training Classifier...")
                clf = build_cat_model(
                    iterations=args.n_est, learning_rate=args.lr,
                    depth=args.cat_depth, l2_leaf_reg=args.cat_l2_leaf_reg,
                    random_seed=args.seed,
                    early_stopping_rounds=args.early_stopping_rounds,
                    task="classification"
                )
                train_pool_bin = Pool(X_train, y_train_bin, weight=w_train)
                valid_pool_bin = Pool(X_valid, y_valid_bin, weight=w_valid)
                clf.fit(train_pool_bin, eval_set=valid_pool_bin)

                print("  Training Regressor...")
                reg = build_cat_model(
                    iterations=args.n_est, learning_rate=args.lr,
                    depth=args.cat_depth, l2_leaf_reg=args.cat_l2_leaf_reg,
                    random_seed=args.seed,
                    early_stopping_rounds=args.early_stopping_rounds,
                    task="regression"
                )
                train_pool_reg = Pool(X_train, y_train, weight=w_train)
                valid_pool_reg = Pool(X_valid, y_valid, weight=w_valid)
                reg.fit(train_pool_reg, eval_set=valid_pool_reg)

                best_iter = reg.best_iteration_
                del X_train

                pred_prob = clf.predict_proba(X_valid)[:, 1]
                pred_val = reg.predict(X_valid)
                pred = pred_prob * pred_val
                del X_valid

                score = _score_valid(pred, y_valid_raw, w_valid)
                print(f"  CV score: {score:.6f}  best_iter(reg): {best_iter}")

                clf.save_model(str(cv_dir / f"fold_{fold_idx}_clf.cbm"))
                reg.save_model(str(cv_dir / f"fold_{fold_idx}_reg.cbm"))
                scores.append(float(score))
                scores_path.write_text(json.dumps({"scores": scores}, indent=2), encoding="utf-8")
                fold_boosters.append((fold_idx, ("cat_mt", str(cv_dir / f"fold_{fold_idx}")), best_iter))
                del train_df, valid_df, y_train, y_valid, w_train, w_valid, clf, reg, train_pool_bin, valid_pool_bin, train_pool_reg, valid_pool_reg
                continue

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
            categorical_feature=[0] if asset_as_cat else None,
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

    if getattr(args, "save_model", False):
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
            elif backend == "xgb_mt":
                # Multi-task xgb saves two files
                fname_clf = f"booster_fold_{fold_idx}_clf.json"
                fname_reg = f"booster_fold_{fold_idx}_reg.json"
                shutil.copyfile(f"{booster_or_path}_clf.json", str(out_dir / fname_clf))
                shutil.copyfile(f"{booster_or_path}_reg.json", str(out_dir / fname_reg))
                # For meta json we can just record the base name
                fname = f"booster_fold_{fold_idx}"
            elif backend == "cat_mt":
                # Multi-task cat saves two files
                fname_clf = f"booster_fold_{fold_idx}_clf.cbm"
                fname_reg = f"booster_fold_{fold_idx}_reg.cbm"
                shutil.copyfile(f"{booster_or_path}_clf.cbm", str(out_dir / fname_clf))
                shutil.copyfile(f"{booster_or_path}_reg.cbm", str(out_dir / fname_reg))
                fname = f"booster_fold_{fold_idx}"
            else:
                fname = f"booster_fold_{fold_idx}.txt"
                booster_or_path.save_model(str(out_dir / fname))
            saved_fold_files.append(fname)

        # Final feature column order = raw + cs + rolling + interaction (matches build_fold_features).
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
        inter_cols = interaction_column_names(interaction_pairs) if interaction_pairs else []
        all_feature_cols = list(raw_feature_cols) + cs_cols + roll_cols + inter_cols
        if asset_as_cat:
            all_feature_cols = ["asset_id"] + all_feature_cols

        meta = {
            "backend": args.backend + ("_mt" if getattr(args, "multi_task", False) else ""),
            "asset_as_categorical": asset_as_cat,
            "feature_columns": all_feature_cols,
            "raw_feature_columns": list(raw_feature_cols),
            "cs_source_columns": cs_source_cols,
            "cs_feature_columns": cs_cols,
            "rolling_source_columns": rolling_source_cols,
            "rolling_feature_columns": roll_cols,
            "rolling_windows": list(rolling_windows),
            "interaction_source_columns": [list(p) for p in interaction_pairs] if interaction_pairs else [],
            "interaction_feature_columns": inter_cols,
            "neutralize_alpha": getattr(args, "neutralize_alpha", 0.0),
            "neutralize_features": drift_features,
            "target_transform": args.target_transform,
            "target_quantile_lut": target_lut,
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

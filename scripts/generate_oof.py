"""Generate out-of-fold predictions for the asset-categorical base models.

Used to evaluate OOF stacking's upside before committing to a full stacking
implementation. For each backend (lgb/xgb/cat), reload the saved fold boosters
and predict on that fold's valid set (the same expanding-window time CV with
embargo=5000 used at training). Saves a single parquet with the three OOF
prediction columns plus time_id/asset_id/weight/target, so we can:

  1. Compute pairwise Pearson correlation of the three OOF predictions.
     If >0.99, stacking upside < 1e-5 -> skip stacking.
  2. Compare Ridge-weighted vs equal-weighted R² on the OOF.

This does NOT retrain — it only runs booster.predict on valid data, so it is
much cheaper than retraining (no GPU, just CPU inference over ~6.6M valid rows
per backend).

Usage:
    python scripts/generate_oof.py --data-root data --partitions 9 \
        --n-folds 5 --embargo 5000 --out out/oof_all.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Ensure the project root (parent of scripts/) is importable for src.*.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_booster(backend: str, path: Path):
    if backend == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor()
        m.load_model(str(path))
        return m
    elif backend == "cat":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(str(path))
        return m
    else:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--partitions", type=int, default=9)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--embargo", type=int, default=5000)
    ap.add_argument("--out", default="out/oof_all.parquet")
    ap.add_argument("--backends", nargs="+", default=["lgb", "xgb", "cat"],
                    help="Backends to generate OOF for (must have strategy_<be>/ with model_meta.json).")
    args = ap.parse_args()

    from src.dataset import load_train
    from src.cv import time_cv_split

    print("Loading train ...")
    frame = load_train(args.data_root, partitions=args.partitions)

    # n_folds / valid_frac / embargo come from the base models' meta, not the
    # CLI, so the OOF folds exactly match training. Read from the first backend.
    first_meta = json.loads(Path(f"strategy_{args.backends[0]}").joinpath("model_meta.json").read_text(encoding="utf-8"))
    n_folds = first_meta.get("n_folds", args.n_folds)
    valid_frac = args.valid_frac
    embargo = args.embargo
    folds = time_cv_split(frame, n_folds=n_folds, valid_frac=valid_frac, embargo=embargo)
    print(f"  {len(folds)} folds (from meta n_folds={n_folds}, embargo={embargo})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate OOF per backend, then join on (time_id, asset_id).
    oof_frames = []
    for be in args.backends:
        be_dir = Path(f"strategy_{be}")
        meta = json.loads((be_dir / "model_meta.json").read_text(encoding="utf-8"))
        # feature_columns already includes asset_id prepended (asset_as_categorical=True).
        feature_cols = list(meta["feature_columns"])
        fold_files = meta.get("fold_files", [])
        print(f"\n=== {be}: {len(fold_files)} folds, {len(feature_cols)} features ===")
        assert len(fold_files) == len(folds), f"fold count mismatch: {len(fold_files)} vs {len(folds)}"

        parts = []
        for fold_idx, (train_lf, valid_lf) in enumerate(folds):
            valid_df = valid_lf.collect()
            X_valid = valid_df.select(feature_cols).to_numpy().astype(np.float32)
            booster = load_booster(be, be_dir / fold_files[fold_idx])
            pred = booster.predict(X_valid)
            print(f"  fold {fold_idx}: valid rows {len(valid_df)}, pred mean {pred.mean():.4f}")
            parts.append(valid_df.select(["time_id", "asset_id", "weight", "target"]).with_columns(
                pl.Series(f"oof_{be}", pred.astype(np.float32))
            ))
            del X_valid, booster
        oof_be = pl.concat(parts, how="vertical_relaxed")
        oof_frames.append(oof_be)

    # Join all backends on (time_id, asset_id). They share identical valid rows.
    # All frames have the same row order (same folds), so just stack the oof column.
    cols = ["time_id", "asset_id", "weight", "target"]
    out = oof_frames[0].select(cols)
    for be, f in zip(args.backends, oof_frames):
        out = out.with_columns(f[f"oof_{be}"])
    out.write_parquet(out_path)
    print(f"\nWrote {out_path}  rows={out.height}  cols={out.columns}")

    # Quick correlation + R² check.
    oof_cols = [f"oof_{be}" for be in args.backends]
    print("\n=== Pairwise Pearson correlation of OOF predictions ===")
    arr = out.select(oof_cols).to_numpy()
    for i in range(len(oof_cols)):
        for j in range(i + 1, len(oof_cols)):
            r = float(np.corrcoef(arr[:, i], arr[:, j])[0, 1])
            print(f"  {oof_cols[i]} vs {oof_cols[j]}: {r:.5f}")

    # Weighted R²: equal-weight vs best-Ridge (closed form on 3 cols).
    y = out["target"].to_numpy().astype(np.float64)
    w = out["weight"].to_numpy().astype(np.float64)
    from src.metrics import weighted_zero_mean_r2

    eq_pred = arr.mean(axis=1)
    eq_r2 = weighted_zero_mean_r2(y, eq_pred, w)
    print(f"\nEqual-weight R2: {eq_r2:.6f}")

    # Per-backend R²
    for i, be in enumerate(args.backends):
        r2 = weighted_zero_mean_r2(y, arr[:, i], w)
        print(f"  {be} R2: {r2:.6f}")

    # Ridge (closed-form weighted ridge regression, intercept=False, alpha=1.0)
    X = arr.astype(np.float64)
    XtW = X.T * w  # (3, n)
    XtWX = XtW @ X + np.eye(X.shape[1]) * 1.0  # ridge
    XtWy = XtW @ y
    coef = np.linalg.solve(XtWX, XtWy)
    ridge_pred = X @ coef
    ridge_r2 = weighted_zero_mean_r2(y, ridge_pred, w)
    print(f"\nRidge coef (alpha=1.0, no intercept): {coef.tolist()}")
    print(f"Ridge-weighted R2: {ridge_r2:.6f}")
    print(f"Ridge - equal-weight: {ridge_r2 - eq_r2:+.6f}  (this is the stacking CV-upside ceiling)")


if __name__ == "__main__":
    main()

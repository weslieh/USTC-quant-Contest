"""Generate out-of-fold predictions for the NN (train_nn.py) model.

Mirrors scripts/generate_oof.py but for the torch fold models. Reuses the SAME
expanding-window time CV split (deterministic from n_folds/valid_frac/embargo +
the same partitions) so NN OOF rows align exactly with the GBDT OOF rows on
(time_id, asset_id). For each fold, load the saved state_dict (booster_fold_{i}.pt),
predict on that fold's valid set (the rows the fold's NN did NOT train on), unscale
by target_std if the NN was trained with --standardize-target, and concatenate.

Output parquet: time_id, asset_id, weight, target, oof_nn  (matches generate_oof.py's
schema so downstream distillation/blending code is uniform).

These OOF predictions are the teacher signal for soft-target distillation: train a
GBDT student on y_soft = (1-lambda)*y + lambda*oof_nn. They are out-of-fold w.r.t.
the NN (each row's prediction came from the fold that held it out) and depend only
on the row's features (row-wise NN, no cross-sample dependency), so using them as a
target component for the GBDT leaks no label.

Usage (GPU):
    python scripts/generate_nn_oof.py --data-root data --partitions 9 \
        --n-folds 5 --embargo 5000 --nn-dir strategy_nn_fixed --out out/oof_nn.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

# Ensure the project root (parent of scripts/) is importable for src.* and train_nn.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_train
from src.cv import time_cv_split
from src.metrics import weighted_zero_mean_r2
from train_nn import TabularNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--partitions", type=int, default=9)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--embargo", type=int, default=5000)
    ap.add_argument("--nn-dir", default="strategy_nn_fixed",
                    help="Directory with model_meta.json + booster_fold_{i}.pt.")
    ap.add_argument("--out", default="out/oof_nn.parquet")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu.")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    nn_dir = Path(args.nn_dir)
    meta = json.loads((nn_dir / "model_meta.json").read_text(encoding="utf-8"))
    raw_feature_cols = list(meta["raw_feature_columns"])
    target_std = float(meta.get("target_std") or 1.0)
    standardize_target = bool(meta.get("standardize_target", False))
    per_asset_specs = [tuple(s) for s in meta.get("per_asset_specs", [])]
    pa_col_names = list(meta.get("per_asset_feature_columns", []))
    fold_files = meta.get("fold_files", [])
    n_features = int(meta["n_features"])

    # Fold split must match training exactly. n_folds/valid_frac/embargo from CLI;
    # time_cv_split is deterministic from these + the same partitions.
    n_folds = args.n_folds
    valid_frac = args.valid_frac
    embargo = args.embargo

    print("Loading train ...")
    frame = load_train(args.data_root, partitions=args.partitions)
    folds = time_cv_split(frame, n_folds=n_folds, valid_frac=valid_frac, embargo=embargo)
    print(f"  {len(folds)} folds (n_folds={n_folds}, valid_frac={valid_frac}, embargo={embargo})")
    assert len(fold_files) == len(folds), \
        f"fold count mismatch: meta has {len(fold_files)} fold_files vs {len(folds)} CV folds"

    # Build per-asset spec feature indices once (raw feature name -> column index).
    if per_asset_specs:
        from src.interactions import build_per_asset_features
        pa_spec_idx = np.asarray(
            [[int(aid), raw_feature_cols.index(fname)]
             for aid, fname in per_asset_specs if fname in raw_feature_cols],
            dtype=np.intp,
        ).reshape(-1, 2)
    else:
        pa_spec_idx = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    for fold_idx, (train_lf, valid_lf) in enumerate(folds):
        valid_df = valid_lf.collect()
        X_valid = np.nan_to_num(valid_df.select(raw_feature_cols).to_numpy().astype(np.float32))
        if pa_spec_idx is not None and pa_col_names:
            valid_df_pa, _ = build_per_asset_features(valid_df, per_asset_specs)
            pa_va = np.nan_to_num(valid_df_pa.select(pa_col_names).to_numpy().astype(np.float32))
            X_valid = np.hstack([X_valid, pa_va])
        aid_va = valid_df["asset_id"].to_numpy().astype(np.int64)
        Xva_t = torch.tensor(X_valid, device=device)
        aidva_t = torch.tensor(aid_va, device=device)

        m = TabularNN(
            n_features=n_features, n_assets=int(meta["n_assets"]),
            n_periodic=int(meta["n_periodic"]), feat_emb_dim=int(meta["feat_emb_dim"]),
            asset_emb_dim=int(meta["asset_emb_dim"]), hidden=int(meta["hidden"]),
            n_blocks=int(meta["n_blocks"]), dropout=float(meta.get("dropout", 0.1)),
        ).to(device)
        state = torch.load(nn_dir / fold_files[fold_idx], map_location=device)
        m.load_state_dict(state)
        m.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, Xva_t.shape[0], 65536):
                preds.append(m(Xva_t[i:i + 65536], aidva_t[i:i + 65536]).cpu().numpy())
        pred = np.concatenate(preds).astype(np.float32)
        # Unscale to the original target scale (matches inference).
        if standardize_target and target_std > 0:
            pred = pred * target_std

        y_va = valid_df["target"].to_numpy().astype(np.float64)
        w_va = valid_df["weight"].to_numpy().astype(np.float64)
        r2 = weighted_zero_mean_r2(y_va, pred.astype(np.float64), w_va)
        vt = sorted(valid_df["time_id"].unique().to_list())
        print(f"  fold {fold_idx}: valid rows {len(valid_df)}, "
              f"time_ids [{vt[0]}..{vt[-1]}] ({len(vt)}), "
              f"pred std {pred.std():.4f}, OOF R2 {r2:.6f}")
        parts.append(valid_df.select(["time_id", "asset_id", "weight", "target"]).with_columns(
            pl.Series("oof_nn", pred)
        ))
        del Xva_t, aidva_t, m, X_valid
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pl.concat(parts, how="vertical_relaxed")
    out.write_parquet(out_path)
    print(f"\nWrote {out_path}  rows={out.height}  cols={out.columns}")

    # Quick weighted R² of the full NN OOF (sanity: should be close to the NN's
    # public-LB for a *fixed* NN, unlike the collapsed NN where CV underestimated).
    y = out["target"].to_numpy().astype(np.float64)
    w = out["weight"].to_numpy().astype(np.float64)
    p = out["oof_nn"].to_numpy().astype(np.float64)
    print(f"NN OOF weighted R2 (over {len(y)} held-out rows): {weighted_zero_mean_r2(y, p, w):.6f}")
    print(f"  (note: OOF covers only the last ~{int(100*args.n_folds*args.valid_frac)}% of "
          f"time_ids — earliest rows are never held out and have no OOF)")


if __name__ == "__main__":
    main()

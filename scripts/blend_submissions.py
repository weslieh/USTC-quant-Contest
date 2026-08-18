"""Blend two or more public-LB prediction CSVs by row_id (weighted average).

For cross-path ensembling where the models can't share a feature spec inside
strategy_ensemble (e.g. an LGB-with-history single model + an old 3-backend-
no-history ensemble). Each input is a (row_id, target) CSV produced by
run_timeseries_api.py. Outputs a blended (row_id, target) CSV for submission.

Usage:
    python scripts/blend_submissions.py \
        --sub out/sub_lgb_hist_l2_50.csv:0.5 \
        --sub out/sub_old_ensemble_0.00310366.csv:0.5 \
        --out out/sub_blend.csv

Weights are normalised. row_id alignment is enforced (mismatch is an error).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_sub(s):
    if ":" in s:
        path, w = s.rsplit(":", 1)
        return path.strip(), float(w)
    return s.strip(), 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub", action="append", required=True,
                   help='"csv_path:weight" (weight optional, default 1). Repeatable.')
    p.add_argument("--out", required=True, help="Output blended CSV path.")
    args = p.parse_args()

    if len(args.sub) < 2:
        raise SystemExit("need at least 2 --sub inputs to blend")

    subs = [parse_sub(s) for s in args.sub]
    total_w = sum(w for _, w in subs)
    if total_w <= 0:
        raise SystemExit(f"total weight must be positive, got {total_w}")

    base = None
    for path, w in subs:
        df = pd.read_csv(path)
        if "row_id" not in df.columns or "target" not in df.columns:
            raise SystemExit(f"{path}: need row_id,target columns, got {list(df.columns)}")
        df = df.sort_values("row_id").reset_index(drop=True)
        if base is None:
            base = df.rename(columns={"target": "target_0"}).copy()
            base["w_0"] = w
            row_ids = base["row_id"].to_numpy()
        else:
            if not np.array_equal(df["row_id"].to_numpy(), row_ids):
                raise SystemExit(f"{path}: row_id mismatch with first submission")
            base[f"target_{len(base.filter(like='target_').columns)}"] = df["target"].to_numpy()
            base[f"w_{len(base.filter(like='w_').columns) - 1}"] = w
        print(f"  {path} weight={w} rows={len(df)}")

    target_cols = [c for c in base.columns if c.startswith("target_")]
    weight_cols = [c for c in base.columns if c.startswith("w_")]
    weights = np.array([base[c].iloc[0] for c in weight_cols], dtype=np.float64)
    weights = weights / weights.sum()
    blended = np.zeros(len(base), dtype=np.float64)
    for tc, wn in zip(target_cols, weights):
        blended += wn * base[tc].to_numpy(dtype=np.float64)

    out_df = pd.DataFrame({"row_id": base["row_id"], "target": blended})
    if not np.isfinite(out_df["target"]).all():
        n_bad = int((~np.isfinite(out_df["target"])).sum())
        out_df.loc[~np.isfinite(out_df["target"]), "target"] = 0.0
        print(f"  WARNING: {n_bad} non-finite predictions set to 0")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} — {len(out_df)} rows, weights={list(weights)}")


if __name__ == "__main__":
    main()

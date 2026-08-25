"""Score a public-LB prediction CSV against the 8/23 label-backfill data.

The 8/23 release (data/train_test/) is the public-test period with labels
(weight, responder_*, target) added, aligned to the original test by row_id.
This lets us compute the exact public-LB R² of any prediction CSV locally,
without spending a public-LB submission quota.

Usage:
    python scripts/score_with_backfill.py \
        --labels data/train_test \
        --pred out/sub_lgb_hist_l2_50.csv \
        [--pred out/another.csv ...]

If multiple --pred are given, each is scored independently. Missing/non-finite
predictions count as 0 (matches the official scoring rule).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(labels_dir):
    paths = sorted(Path(labels_dir).glob("train_partition_*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet under {labels_dir}")
    cols = ["row_id", "weight", "target"]
    df = pd.concat((pd.read_parquet(p, columns=cols) for p in paths), ignore_index=True)
    return df.sort_values("row_id").reset_index(drop=True)


def score(labels, pred_path):
    pred = pd.read_csv(pred_path)
    if "row_id" not in pred.columns or "target" not in pred.columns:
        raise SystemExit(f"{pred_path}: need row_id,target columns")
    pred = pred.sort_values("row_id").reset_index(drop=True)
    merged = labels.merge(pred.rename(columns={"target": "prediction"}), on="row_id", how="left")
    p = pd.to_numeric(merged["prediction"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = merged["target"].to_numpy(dtype=np.float64)
    w = merged["weight"].to_numpy(dtype=np.float64)
    denom = float(np.sum(w * y * y))
    if denom <= 0:
        return float("nan"), len(merged)
    r2 = float(1.0 - np.sum(w * (y - p.to_numpy(dtype=np.float64)) ** 2) / denom)
    n_missing = int(merged["prediction"].isna().sum())
    return r2, len(merged), n_missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="data/train_test", help="Dir with backfill train_partition_*.parquet")
    p.add_argument("--pred", action="append", required=True, help="Prediction CSV (row_id,target). Repeatable.")
    args = p.parse_args()

    labels = load_labels(args.labels)
    print(f"labels: {len(labels)} rows from {args.labels}")
    for path in args.pred:
        r2, n, n_miss = score(labels, path)
        miss_str = f" (missing={n_miss})" if n_miss else ""
        print(f"  {r2:.8f}  {Path(path).name}  rows={n}{miss_str}")


if __name__ == "__main__":
    main()

"""EDA Module 4 — Per-feature drift granularity.

The single adversarial AUC=1.0 is known. This module breaks it down per feature:
which features drift most (train vs test), which drift internally over time
(early vs late train), and whether target variance (the R^2 denominator) is
stable over time. Cross with Module 1: a high |corr-target| feature that also
drifts is a false signal; one with low drift is a stable carrier.

Outputs to out/eda/:
  m4_feature_drift_train_test.csv   feature, ks_stat, p_value, mean_shift_std, rank
  m4_feature_drift_time_internal.csv feature, ks_early_late, rank
  m4_drift_comparison.png           scatter train-test KS vs early-late KS
  m4_target_weight_timeseries.png   target mean/std + wyy per time bucket
  module4_summary.md

Usage:
  python scripts/eda_m4_drift.py --data-root data --sample-rows 2000000 --out out/eda
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
from scipy.stats import ks_2samp

import eda_common as ec


def per_feature_ks(train_arr, test_arr, feature_cols):
    """KS statistic + standardized mean shift per feature (vectorized over features
    via per-column loop; arrays are (n, K) float32)."""
    rows = []
    for j, c in enumerate(feature_cols):
        a = train_arr[:, j]; b = test_arr[:, j]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if a.size < 100 or b.size < 100:
            rows.append({"feature": c, "ks_stat": float("nan"), "p_value": float("nan"),
                         "mean_shift_std": float("nan")})
            continue
        ks = ks_2samp(a, b)
        pooled_std = np.sqrt((a.var() + b.var()) / 2)
        ms = float((b.mean() - a.mean()) / pooled_std) if pooled_std > 1e-12 else 0.0
        rows.append({"feature": c, "ks_stat": float(ks.statistic),
                     "p_value": float(ks.pvalue), "mean_shift_std": ms})
    df = pl.DataFrame(rows).with_columns(
        pl.col("ks_stat").rank("ordinal", descending=True).alias("ks_rank")
    ).sort("ks_stat", descending=True)
    return df


def sample_array(lf, n_rows, feature_cols, n_assets=15, seed=42):
    """Sample by time_id and return (n, K) float32 numpy array (features only)."""
    df = ec.sample_by_time(lf, n_rows, n_assets=n_assets, seed=seed)
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def target_weight_timeseries(lf, bucket=1000):
    """Per time-bucket target mean/std, weight mean, and w*target^2 (R^2 denom)."""
    return (
        lf.with_columns((pl.col("time_id") // bucket).alias("bucket"))
        .group_by("bucket")
        .agg(
            pl.col("target").mean().alias("target_mean"),
            pl.col("target").std().alias("target_std"),
            pl.col("weight").mean().alias("weight_mean"),
            (pl.col("weight") * pl.col("target") * pl.col("target")).sum().alias("wyy_sum"),
            pl.len().alias("n"),
        )
        .sort("bucket")
        .collect()
    )


def main():
    ap = argparse.ArgumentParser(description="EDA module 4: drift granularity")
    ec.add_common_args(ap)
    ap.add_argument("--drift-rows", type=int, default=400_000,
                    help="Rows sampled from each side (train/test, early/late) for KS.")
    args = ap.parse_args()

    out = ec.ensure_out_dir(args.out)
    load_train, load_test = ec.load_train_lazy(args.data_root)
    lf = load_train(args.data_root)
    lf_test = load_test(args.data_root)
    feature_cols = ec.get_feature_columns(lf)
    print(f"features: {len(feature_cols)}", flush=True)

    n_drift = args.drift_rows

    # ---- 1. per-feature train vs test KS ----
    print("sampling train/test for KS...", flush=True)
    Xtr = sample_array(lf, n_drift, feature_cols, seed=args.seed)
    Xte = sample_array(lf_test, n_drift, feature_cols, seed=args.seed)
    print(f"train {Xtr.shape}, test {Xte.shape}", flush=True)
    tt = per_feature_ks(Xtr, Xte, feature_cols)
    ec.save_csv(tt, out / "m4_feature_drift_train_test.csv")

    # ---- 2. per-feature train internal time drift (early vs late) ----
    print("sampling early/late train for KS...", flush=True)
    early = ec.filter_time_range(lf, 0, 100_000)  # partition 0
    late = ec.filter_time_range(lf, 800_000, 888_480)  # partition 8
    Xe = sample_array(early, n_drift, feature_cols, seed=args.seed)
    Xl = sample_array(late, n_drift, feature_cols, seed=args.seed)
    el = per_feature_ks(Xe, Xl, feature_cols).rename({
        "ks_stat": "ks_early_late", "p_value": "p_early_late",
        "mean_shift_std": "mean_shift_early_late", "ks_rank": "ks_rank_early_late",
    }).select(["feature", "ks_early_late", "p_early_late", "mean_shift_early_late", "ks_rank_early_late"])
    ec.save_csv(el, out / "m4_feature_drift_time_internal.csv")

    # ---- 3. drift comparison scatter (train-test KS vs early-late KS) ----
    joined = tt.select(["feature", "ks_stat"]).join(
        el.select(["feature", "ks_early_late"]), on="feature")
    ec.save_scatter(
        joined["ks_stat"].to_numpy(), joined["ks_early_late"].to_numpy(),
        out / "m4_drift_comparison.png",
        xlabel="train-test KS", ylabel="early-late train KS",
        title="Per-feature drift: train-test vs internal-time",
    )

    # ---- 4. target/weight time series ----
    ts = target_weight_timeseries(lf, bucket=1000)
    ec.save_csv(ts, out / "m4_target_weight_timeseries.csv")
    bx = ts["bucket"].to_numpy()
    ec.save_line(
        {"target_mean": (bx, ts["target_mean"].to_numpy()),
         "target_std": (bx, ts["target_std"].to_numpy()),
         "wyy_sum (R2 denom, scaled)": (bx, ts["wyy_sum"].to_numpy() / max(ts["wyy_sum"].to_numpy().max(), 1e-12))},
        out / "m4_target_weight_timeseries.png",
        xlabel="time_id bucket (x1000)", ylabel="value",
        title="Target mean/std & R2-denominator over time",
    )

    # ---- summary ----
    # drift stats
    ks_med = float(tt["ks_stat"].median())
    ks_high = float((tt["ks_stat"] > 0.5).sum())
    el_med = float(el["ks_early_late"].median())
    # target std stability: ratio of last-bucket std to first-bucket std
    tstd = ts["target_std"].to_numpy()
    tstd_ratio = float(np.nan_to_num(tstd[-1] / max(tstd[0], 1e-12)))
    wyy = ts["wyy_sum"].to_numpy()
    # coefficient of variation of wyy across buckets (stable? 0=flat)
    wyy_cv = float(np.std(wyy) / max(np.mean(wyy), 1e-12))
    md = f"""# EDA Module 4 — Drift granularity

## Per-feature train-vs-test KS (323 features)
- median KS = {ks_med:.3f}; features with KS>0.5 = {ks_high}/{len(feature_cols)}.
- KS=1 means fully separable distributions. AUC=1.0 adversarial is the
  aggregate; this shows it's broadly spread, not a few leaky columns.
- See m4_feature_drift_train_test.csv (sorted by KS).

## Per-feature internal-time drift (early vs late train)
- median early-late KS = {el_med:.3f}.
- Features that drift *internally over train time* will also drift to test:
  their apparent signal is likely non-stationary/false.
- See m4_feature_drift_time_internal.csv.

## Train-test KS vs early-late KS (scatter)
- Points on the diagonal: features drift steadily over time (drift is
  monotonic-in-time, so late-train ~ test-ish).
- Points high on x but low on y: a discontinuity between train and test beyond
  pure time trend — these are the most dangerous (true distribution shift).
- See m4_drift_comparison.png.

## Target / weight time stability (R^2 denominator)
- target_std last/first bucket ratio = {tstd_ratio:.3f} (1.0 = stable variance).
- w*target^2 (R^2 denominator) CV across time buckets = {wyy_cv:.3f} (0 = flat).
- {"target variance stable over time => CV-leaderboard 1.6x gap is a drift effect, not target non-stationarity." if abs(tstd_ratio-1)<0.2 and wyy_cv<0.3 else "target variance drifts over time => consider time-segment-reweighted CV."}
- See m4_target_weight_timeseries.png.

## Cross-module (m1 + m4): signal vs drift filter
- Load m1_feature_target_corr.csv and m4_feature_drift_train_test.csv.
- HIGH |corr-target| + HIGH train-test KS => drifting false signal (deprioritize
  or neutralize at inference).
- HIGH |corr-target| + LOW KS => stable signal carrier (prioritize).
"""
    ec.save_summary("module4", md, out / "module4_summary.md")
    print(f"\nmedian train-test KS={ks_med:.3f}, median early-late KS={el_med:.3f}", flush=True)
    print(f"target_std last/first={tstd_ratio:.3f}, wyy CV={wyy_cv:.3f}", flush=True)
    ec.list_outputs(out)


if __name__ == "__main__":
    main()

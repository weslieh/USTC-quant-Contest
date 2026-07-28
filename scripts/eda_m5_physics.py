"""EDA Module 5 — Feature physical-type reverse-engineering.

Features are anonymized, but the competition host confirmed some are return-like
and the target is a return/risk composite. Physical type can be inferred from
statistical behavior, combining multiple cues:
  - distribution shape (from m1): heavy-tailed return-like, right-skew vol-like, ...
  - per-asset stability: is the feature a near-constant per-asset "level" (price-like)
    or does it vary within an asset?
  - magnitude/range: bounded [0,1] (ratio/probability), large positive (price level),
    near-zero symmetric (return), sparse non-negative (volume/event).
  - drift (from m4): how much it shifts train->test.
  - target/responder correlation (from m1/m2).

Outputs out/eda/feature_physics.csv: per-feature physical-type label + the cues.
This is the basis for physically-motivated feature×feature interactions (e.g.
return / volatility = Sharpe-like), to replace the failed blind top-K pairwise.

Usage:
  python scripts/eda_m5_physics.py --data-root data --out out/eda
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

import eda_common as ec


def infer_physics(moments, per_asset_stats, drift, target_corr, clusters):
    """Assign a physical-type label per feature from the cue table.

    moments: DataFrame from m1 (column, mean, std, skew, kurt, q01, q50, q99, min, max, category)
    per_asset_stats: DataFrame (column, within_asset_std_ratio, mean_of_asset_means, std_of_asset_means)
       within_asset_std_ratio = median within-asset std / global std  (low => near-constant per asset)
       std_of_asset_means = std of per-asset means (high => asset-level level feature)
    drift: DataFrame (column, ks_stat)
    target_corr: DataFrame (feature, abs_corr)
    clusters: DataFrame (feature, cluster_id)
    """
    m = moments.with_columns(pl.col("column").alias("feature"))
    pa = per_asset_stats.rename({"column": "feature"})
    dr = drift.select(["feature", "ks_stat"]).rename({"ks_stat": "drift_ks"})
    tc = target_corr.select(["feature", "abs_corr"]).rename({"abs_corr": "abs_corr_target"})
    df = m.join(pa, on="feature", how="left").join(dr, on="feature", how="left") \
         .join(tc, on="feature", how="left").join(clusters, on="feature", how="left")

    # Heuristic physical-type assignment. Order matters: most specific first.
    def label(row):
        mean = row["mean"]; std = row["std"]; skew = row["skew"]; kurt = row["kurt"]
        q01 = row["q01"]; q99 = row["q99"]; mn = row["min"]; mx = row["max"]
        within_ratio = row["within_asset_std_ratio"]  # median within-asset std / global std
        asset_mean_std = row["std_of_asset_means"]
        if any(v is None for v in [mean, std, skew, kurt, q01, q99, mn, mx]):
            return "unknown"
        # price_level: near-constant per asset (within_ratio tiny), positive magnitude,
        # asset means differ. e.g. feature_304 (9.49, within std 1e-6).
        if within_ratio is not None and within_ratio < 0.02 and mean > 1.0 and asset_mean_std is not None and asset_mean_std < 0.01 * abs(mean):
            return "price_level"
        # ratio_probability: bounded in [0,1] (allow tiny overshoot from float), mean in (0,1)
        if mn >= -0.01 and mx <= 1.01 and 0.0 < mean < 1.0 and std < 0.5:
            return "ratio_probability"
        # volume_event: non-negative, very heavy right tail, mostly near zero (sparse)
        if mn >= -1e-6 and skew is not None and skew > 5 and kurt > 50 and q99 is not None and q99 > 0 and abs(mean) < q99 * 0.5:
            return "volume_event"
        # volatility_like: non-negative, right-skew, moderate tail (vol/size)
        if mn >= -1e-6 and skew is not None and 0.5 < skew <= 5 and mean > 0:
            return "volatility_like"
        # return_like: near-zero mean, symmetric (|skew| small), heavy tail
        if abs(mean) < 0.05 * max(std, 1e-6) and abs(skew) < 1.5 and kurt > 6:
            return "return_like"
        # near_gaussian symmetric
        if abs(skew) < 0.5 and 2 < kurt < 6:
            return "symmetric_gaussian_like"
        if abs(skew) < 0.5:
            return "symmetric_other"
        return "other"

    labels = [label(r) for r in df.iter_rows(named=True)]
    df = df.with_columns(pl.Series("physics_type", labels))
    # select + order output columns
    out_cols = ["feature", "physics_type", "cluster_id", "mean", "std", "skew", "kurt",
                "q01", "q50", "q99", "min", "max",
                "within_asset_std_ratio", "std_of_asset_means", "drift_ks",
                "abs_corr_target"]
    return df.select([c for c in out_cols if c in df.columns]).sort("physics_type", "feature")


def per_asset_stability(lf, feature_cols, n_sample_assets_rows=400_000):
    """For each feature, compute median(within-asset std)/global std and std of per-asset means.

    Low within_ratio + high asset_mean_std => a per-asset "level" feature (price-like).
    Computed on a time-stratified sample to bound memory; stable enough for labeling.
    """
    # per-asset mean and std (streaming via group_by on lazy, chunked to avoid the
    # 323-expr plan blowup). Use a sample for the std (means are exact-enough).
    sample = ec.sample_by_time(lf, n_sample_assets_rows, seed=42)
    means = sample.group_by("asset_id").agg([pl.col(c).mean().alias(c) for c in feature_cols])
    stds = sample.group_by("asset_id").agg([pl.col(c).std().alias(c) for c in feature_cols])
    m_arr = means.select(feature_cols).to_numpy().astype(np.float64)  # (15, 323)
    s_arr = stds.select(feature_cols).to_numpy().astype(np.float64)   # (15, 323)
    global_std = sample.select([pl.col(c).std().alias(c) for c in feature_cols]).to_numpy()[0]
    # median within-asset std / global std
    within_ratio = np.nanmedian(s_arr, axis=0) / np.where(global_std < 1e-12, 1.0, global_std)
    std_of_means = m_arr.std(axis=0)
    return pl.DataFrame({
        "column": feature_cols,
        "within_asset_std_ratio": within_ratio.tolist(),
        "std_of_asset_means": std_of_means.tolist(),
    })


def main():
    ap = argparse.ArgumentParser(description="EDA module 5: feature physical-type inference")
    ec.add_common_args(ap)
    args = ap.parse_args()
    out = ec.ensure_out_dir(args.out)

    # Reuse existing EDA outputs where possible.
    moments = pl.read_csv(out / "m1_feature_distribution_profile.csv")
    clusters = pl.read_csv(out / "m1_feature_clusters.csv")
    target_corr = pl.read_csv(out / "m1_feature_target_corr.csv")
    drift = pl.read_csv(out / "m4_feature_drift_train_test.csv")

    load_train, _ = ec.load_train_lazy(args.data_root)
    lf = load_train(args.data_root)
    feature_cols = ec.get_feature_columns(lf)
    print(f"features: {len(feature_cols)}", flush=True)

    print("computing per-asset stability (sample)...", flush=True)
    pa = per_asset_stability(lf, feature_cols)
    ec.save_csv(pa, out / "m5_per_asset_stability.csv")

    physics = infer_physics(moments, pa, drift, target_corr, clusters)
    ec.save_csv(physics, out / "feature_physics.csv")

    # summary
    counts = physics.group_by("physics_type").agg(pl.len().alias("n")).sort("n", descending=True)
    md = f"""# EDA Module 5 — Feature physical-type inference

Host confirmed some features are return-like and the target is a return/risk
composite. Physical type inferred from: distribution shape (m1), per-asset
stability (within-asset std / global std; std of per-asset means), magnitude,
drift (m4), target correlation (m1), correlation cluster (m1).

## Type counts
{ec.md_table(counts)}

## Cues used
- **price_level**: near-constant per asset (within_ratio<0.02), positive magnitude,
  per-asset means barely differ within an asset but the level is asset-specific-ish.
  e.g. feature_304 (9.491853, within-asset std 1e-6).
- **ratio_probability**: bounded in [0,1], mean in (0,1). e.g. feature_310 (~1.0).
- **volume_event**: non-negative, very heavy right tail (skew>5, kurt>50), mostly near zero.
- **volatility_like**: non-negative, right-skew (0.5<skew<=5), positive mean.
- **return_like**: near-zero mean, symmetric (|skew|<1.5), heavy tail (kurt>6).
- **symmetric_gaussian_like / symmetric_other / other**: remaining.

## Full table
{ec.md_table(physics, max_rows=323)}

## Use
Physically-motivated cross-type interactions (e.g. return_like / volatility_like
= Sharpe-like; price_level differences = log-return-like) to replace the failed
blind top-K pairwise (which dropped 10% on single-partition AV-CV). Within-row,
drift-safe. See feature_physics.csv for per-feature labels.
"""
    ec.save_summary("module5", md, out / "module5_summary.md")
    print("\nphysics type counts:", flush=True)
    print(counts.to_pandas().to_string(index=False))
    ec.list_outputs(out)


if __name__ == "__main__":
    main()

"""EDA Module 1 — Feature structure + physical-type reverse-engineering.

Features are officially anonymized (no mapping to real market quantities), so
the only way to guess physical type is from distribution *shape*: heavy-tailed
return-like (high kurtosis), right-skew vol-like (positive skew), near-Gaussian
price/volume-like. Plus feature-feature correlation clustering reveals hidden
feature groups, and feature-target weighted correlation ranks the signal carriers.

Outputs to out/eda/:
  m1_feature_corr_heatmap.png          323x323 corr, ordered by cluster
  m1_feature_dendrogram.png
  m1_feature_clusters.csv              feature, cluster_id
  m1_feature_distribution_profile.csv  feature, mean, std, skew, kurt, qs, category
  m1_feature_target_corr.csv           323 rows, weighted corr with target, sorted
  module1_summary.md

Usage:
  python scripts/eda_m1_features.py --data-root data --sample-rows 2000000 --out out/eda
  python scripts/eda_m1_features.py --data-root data --sample-rows 0 --out out/eda   # cloud full
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

import eda_common as ec


def classify_shape(skew, kurt):
    """Coarse physical-type guess from distribution shape."""
    if kurt is None or skew is None or not np.isfinite(kurt) or not np.isfinite(skew):
        return "unknown"
    if kurt > 6:
        return "heavy_tailed_return_like"
    if skew > 1:
        return "right_skew_vol_like"
    if skew < -1:
        return "left_skew"
    if abs(skew) < 0.5 and 2 < kurt < 4:
        return "near_gaussian"
    if abs(skew) < 0.5:
        return "symmetric"
    return "moderate_skew"


def main():
    ap = argparse.ArgumentParser(description="EDA module 1: feature structure")
    ec.add_common_args(ap)
    args = ap.parse_args()

    out = ec.ensure_out_dir(args.out)
    load_train, _ = ec.load_train_lazy(args.data_root)
    lf = load_train(args.data_root)
    feature_cols = ec.get_feature_columns(lf)
    print(f"features: {len(feature_cols)}", flush=True)

    # ---- 1. 323x323 feature correlation (sampled) ----
    # 323x323 corr only needs ~500k rows (Pearson SE ~0.0014); a 2M-row float64
    # array is 4.5GB + corrcoef copies => OOM on 32GB. Cap at 500k, use float32.
    corr_rows = min(args.sample_rows, 500_000) if args.sample_rows > 0 else 0
    sample = ec.sample_by_time(lf, corr_rows, seed=args.seed)
    print(f"sample rows for corr/moments: {sample.height} (capped at 500k)", flush=True)
    X = sample.select(feature_cols).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    corr = ec.corr_matrix(X, assert_shape=(len(feature_cols), len(feature_cols)))

    # ---- 2. feature clustering ----
    clusters = None
    ordered = list(range(len(feature_cols)))
    try:
        from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
        from scipy.spatial.distance import squareform
        d = np.clip(1 - np.abs(corr), 0, 1)
        np.fill_diagonal(d, 0)
        cond = squareform(d, checks=False)
        Z = linkage(cond, method="average")
        labels_cl = fcluster(Z, t=0.7, criterion="distance")
        clusters = labels_cl
        # order by cluster for a blocky heatmap
        order = np.argsort(labels_cl, kind="stable")
        ordered = order.tolist()
        ec.save_dendrogram(Z, feature_cols, out / "m1_feature_dendrogram.png",
                           title="Feature dendrogram (1-|corr|)")
        ec.save_csv(pl.DataFrame({"feature": feature_cols, "cluster_id": labels_cl.tolist()}),
                    out / "m1_feature_clusters.csv")
    except Exception as e:
        print(f"  [skip clustering] {e}", flush=True)

    ec.save_corr_heatmap(corr[np.ix_(ordered, ordered)],
                         [feature_cols[i] for i in ordered],
                         out / "m1_feature_corr_heatmap.png",
                         title="Feature correlation (cluster-ordered)")

    # ---- 3. feature distribution profile (on the materialized sample; full-scan
    #      multi-quantile exprs blow up polars on 13.2M rows, but a 2M-row sample
    #      is plenty for descriptive moments) ----
    moments = ec.stream_col_moments(sample, feature_cols, chunk=64)
    moments = moments.with_columns(
        pl.struct(["skew", "kurt"]).map_elements(
            lambda r: classify_shape(r["skew"], r["kurt"]), return_dtype=pl.Utf8
        ).alias("category")
    )
    ec.save_csv(moments, out / "m1_feature_distribution_profile.csv")

    # ---- 4. feature-target weighted correlation (streaming, full precision) ----
    fc = ec.stream_weighted_corr(lf, feature_cols, target="target", weight="weight")
    rows = sorted(((c, v) for c, v in fc.items()), key=lambda kv: abs(kv[1]), reverse=True)
    ftc = pl.DataFrame({
        "feature": [c for c, _ in rows],
        "weighted_corr_target": [v for _, v in rows],
        "abs_corr": [abs(v) for _, v in rows],
    })
    ec.save_csv(ftc, out / "m1_feature_target_corr.csv")

    # ---- summary ----
    n_clusters = int(np.max(clusters)) if clusters is not None else 0
    cat_counts = moments.group_by("category").agg(pl.len().alias("n")).sort("n", descending=True)
    top10 = ftc.head(10)
    # how concentrated is the signal? share of |corr| in top-20 vs all
    abs_all = np.array([abs(v) for v in fc.values() if np.isfinite(v)])
    top20_share = float(abs_all[np.argsort(abs_all)[::-1][:20]].sum() / max(abs_all.sum(), 1e-12))
    md = f"""# EDA Module 1 — Feature structure

## Feature clustering
- {n_clusters} clusters at distance threshold 0.7 (1-|corr|).
- A few tight clusters holding most features => features are redundant; candidate
  for within-cluster PCA decorrelation (keep top 1-2 PCs per cluster, lower the
  tree's effective dimension, may reduce drift overfitting).
- See m1_feature_corr_heatmap.png, m1_feature_dendrogram.png, m1_feature_clusters.csv.

## Distribution shape -> physical-type guess
{ec.md_table(cat_counts)}

- heavy_tailed_return_like (kurt>6): likely return/ratio features.
- right_skew_vol_like (skew>1): likely volatility/size features.
- near_gaussian (|skew|<0.5, 2<kurt<4): likely price/volume-level features.
- See m1_feature_distribution_profile.csv.

## Feature-target weighted correlation
- Top 20 features carry **{top20_share:.1%}** of total |corr with target|.
- {"Signal concentrated: restrict interaction/discovery to top ~20, not top-100." if top20_share > 0.4 else "Signal diffuse: many weak features."}
- Top 10:
{ec.md_table(top10)}

## Cross-module use
- Feed the feature clusters + physical-type categories into Module 3's per-asset
  importance interpretation and into future interaction engineering (prefer
  cross-type pairs e.g. return x volatility over same-type pairs).
- Cross with Module 4 drift: a high |corr-target| feature that also has high
  train-test KS is a drifting false signal; one with low KS is a stable carrier.
"""
    ec.save_summary("module1", md, out / "module1_summary.md")
    print(f"\nn_clusters={n_clusters}, top20 |corr| share={top20_share:.1%}", flush=True)
    ec.list_outputs(out)


if __name__ == "__main__":
    main()

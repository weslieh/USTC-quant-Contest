"""EDA Module 3 — Per-asset differences (the top-priority model direction basis).

asset_id categorical is the ONLY direction that survived (+2.4%). This module
quantifies *how* assets differ, to decide between:
  - single shared model + asset_id categorical (current best) and深耕 per-asset
    target transforms, if top features are universal across assets (high Jaccard)
  - per-asset feature selection + partial-sharing cluster models, if top features
    differ by asset (low Jaccard)

Outputs to out/eda/:
  m3_asset_overview.csv              15 rows: n, time_span, target_mean/std, wyy_sum
  m3_asset_target_corr_heatmap.png   15x15 same-time_id target correlation
  m3_asset_feature_dissim_heatmap.png 15x15 feature-mean dissimilarity
  m3_per_asset_importance.csv        15 x 323 LGB gain (averaged over 2 seeds)
  m3_topk_jaccard_heatmap.png        15x15 top-20 feature overlap
  m3_asset_clustering_dendrogram.png
  module3_summary.md

Usage:
  python scripts/eda_m3_perasset.py --data-root data --sample-rows 2000000 --out out/eda
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

import eda_common as ec


N_ASSETS = 15


def per_asset_overview(lf):
    """n, time_span, target_mean, target_std, wyy_sum (R^2 zero baseline) per asset."""
    return ec.weighted_r2_zero_baseline(lf, by="asset_id").with_columns(
        (pl.col("wyy_sum") / pl.col("wyy_sum").sum()).alias("wyy_share"),
    )


def asset_target_corr_matrix(sample):
    """15x15 same-time_id target correlation via pivot to (n_time_id, 15)."""
    piv = (
        sample.select(["time_id", "asset_id", "target"])
        .pivot("asset_id", index="time_id", values="target")
        .sort("time_id")
    )
    asset_cols = [c for c in piv.columns if c != "time_id"]
    M = piv.select(asset_cols).to_numpy().astype(np.float64)
    # drop rows with any NaN (incomplete time_id slices) for corr
    M = M[~np.isnan(M).any(axis=1)]
    assert M.shape[1] == N_ASSETS, f"expected {N_ASSETS} asset cols, got {M.shape[1]}"
    corr = ec.corr_matrix(M, assert_shape=(N_ASSETS, N_ASSETS))
    return corr, asset_cols


def asset_feature_mean_dissim(lf, feature_cols, chunk=64):
    """15x15 normalized feature-mean dissimilarity: ||mean_i - mean_j|| / global_std.

    Means (per asset) and global stds are computed in column chunks to avoid
    polars' expression-plan blowup on a 323-expr select over 13.2M rows.
    """
    n_assets = 15
    means = np.zeros((n_assets, len(feature_cols)), dtype=np.float64)
    stds = np.zeros(len(feature_cols), dtype=np.float64)
    for start in range(0, len(feature_cols), chunk):
        batch = feature_cols[start:start + chunk]
        m = (
            lf.group_by("asset_id")
            .agg([pl.col(c).mean().alias(c) for c in batch])
            .sort("asset_id")
            .collect()
            .select(batch)
            .to_numpy()
            .astype(np.float64)
        )
        s = lf.select([pl.col(c).std().alias(c) for c in batch]).collect().to_numpy()[0].astype(np.float64)
        means[:, start:start + len(batch)] = m
        stds[start:start + len(batch)] = s
    stds = np.where(stds < 1e-12, 1.0, stds)
    means_n = means / stds  # (15, 323) normalized
    diff = means_n[:, None, :] - means_n[None, :, :]
    dissim = np.linalg.norm(diff, axis=2)
    return dissim


def per_asset_importance(lf, feature_cols, per_asset_rows=150_000, seeds=(0, 1)):
    """Train a light LGB per asset on a sample; record feature_importances_,
    averaged over ``seeds``. Returns (15, 323) gain and the asset id order.

    Only features in the top-20 of BOTH seeds are considered robust per asset
    (callers compute Jaccard from the averaged importance to keep it simple,
    but the 2-seed average already denoises).
    """
    from src.model import build_model
    asset_ids = sorted(lf.select(pl.col("asset_id").unique()).collect().to_numpy().ravel().tolist())
    gain = np.zeros((len(asset_ids), len(feature_cols)), dtype=np.float64)
    for ai, aid in enumerate(asset_ids):
        sub = lf.filter(pl.col("asset_id") == aid)
        # sample within the asset
        df = ec.sample_by_time(sub, per_asset_rows, n_assets=1, seed=42)
        if df.height < 2000:
            print(f"  asset {aid}: only {df.height} rows, skip importance", flush=True)
            continue
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = df["target"].to_numpy().astype(np.float32)
        w = df["weight"].to_numpy().astype(np.float32)
        mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
        X, y, w = X[mask], y[mask], w[mask]
        g_acc = np.zeros(len(feature_cols), dtype=np.float64)
        n_ok = 0
        for s in seeds:
            m = build_model(n_estimators=300, learning_rate=0.05, num_leaves=31,
                            min_child_samples=200, random_state=s)
            m.fit(X, y, sample_weight=w)
            g_acc += m.feature_importances_
            n_ok += 1
        gain[ai] = g_acc / max(1, n_ok)
        print(f"  asset {aid}: {df.height} rows, top5 feat importance ok", flush=True)
    return gain, asset_ids


def topk_jaccard(gain, k=20):
    """15x15 Jaccard of top-k feature sets per asset pair."""
    n = gain.shape[0]
    topsets = [set(np.argsort(gain[i])[::-1][:k].tolist()) for i in range(n)]
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(topsets[i] & topsets[j])
            union = len(topsets[i] | topsets[j])
            J[i, j] = inter / union if union > 0 else 0.0
    return J


def main():
    ap = argparse.ArgumentParser(description="EDA module 3: per-asset differences")
    ec.add_common_args(ap)
    ap.add_argument("--importance-rows", type=int, default=150_000,
                    help="Rows per asset for LGB importance (0 = full per-asset).")
    args = ap.parse_args()

    out = ec.ensure_out_dir(args.out)
    load_train, _ = ec.load_train_lazy(args.data_root)
    lf = load_train(args.data_root)
    feature_cols = ec.get_feature_columns(lf)
    print(f"features: {len(feature_cols)}", flush=True)

    # ---- 1. per-asset overview (streaming, full precision) ----
    overview = per_asset_overview(lf)
    ec.save_csv(overview, out / "m3_asset_overview.csv")

    # ---- 2. 15x15 asset target correlation ----
    sample = ec.sample_by_time(lf, args.sample_rows, seed=args.seed)
    print(f"sample rows for asset corr: {sample.height}", flush=True)
    atc, asset_cols = asset_target_corr_matrix(sample)
    ec.save_corr_heatmap(atc, [f"a{c}" for c in asset_cols],
                         out / "m3_asset_target_corr_heatmap.png",
                         title="Same-time_id target correlation across assets")

    # ---- 3. per-asset feature mean dissimilarity ----
    dissim = asset_feature_mean_dissim(lf, feature_cols)
    ec.save_corr_heatmap(dissim, [f"a{i}" for i in range(N_ASSETS)],
                         out / "m3_asset_feature_dissim_heatmap.png",
                         title="Feature-mean dissimilarity (normalized)", vmin=0, vmax=None, cmap="viridis")

    # ---- 4. per-asset feature importance + top-k Jaccard ----
    imp_rows = args.importance_rows if args.importance_rows > 0 else 0
    gain, asset_ids = per_asset_importance(lf, feature_cols, per_asset_rows=imp_rows)
    imp_df = pl.DataFrame(gain, schema=[f"asset_{a}" for a in asset_ids])
    # transpose so rows=asset, cols=feature for readability
    imp_t = pl.DataFrame(
        {"asset_id": asset_ids,
         **{feature_cols[j]: gain[:, j].tolist() for j in range(len(feature_cols))}}
    )
    ec.save_csv(imp_t, out / "m3_per_asset_importance.csv", note="15 x 323 gain")

    J = topk_jaccard(gain, k=20)
    ec.save_corr_heatmap(J, [f"a{i}" for i in range(N_ASSETS)],
                         out / "m3_topk_jaccard_heatmap.png",
                         title="Top-20 feature Jaccard across assets", vmin=0, vmax=1, cmap="viridis")

    # ---- 5. asset clustering (on target correlation distance) ----
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        dist = np.clip(1 - atc, 0, 2)
        np.fill_diagonal(dist, 0)
        cond = squareform(dist, checks=False)
        Z = linkage(cond, method="average")
        ec.save_dendrogram(Z, [f"a{i}" for i in range(N_ASSETS)],
                           out / "m3_asset_clustering_dendrogram.png",
                           title="Asset clustering (1 - target corr)")
    except Exception as e:
        print(f"  [skip dendrogram] {e}", flush=True)

    # ---- summary ----
    jaccard_mean = float(J[~np.eye(N_ASSETS, dtype=bool)].mean())
    overview_pd = overview.to_pandas()
    # rank assets by wyy_share
    overview_sorted = overview.sort("wyy_sum", descending=True)
    md = f"""# EDA Module 3 — Per-asset differences

## Per-asset overview (sorted by wyy_share = share of R^2 zero-prediction denominator)
{overview_sorted.to_pandas().to_markdown(index=False)}

## 15x15 same-time_id target correlation
- Reveals which assets co-move (shared generating process).
- See m3_asset_target_corr_heatmap.png.

## Top-20 feature Jaccard across assets
- Mean off-diagonal Jaccard = **{jaccard_mean:.3f}**
- Decision:
  - mean > 0.6 => top features are universal across assets; asset differences
    are in coefficients/split-points, NOT feature selection. => stick with the
    single shared model + asset_id categorical (current best) and深耕 per-asset
    target transforms / thresholds.
  - mean < 0.3 => top features differ by asset. => per-asset feature selection
    + partial-sharing cluster models (group assets by the 15x15 clustering,
    train one model per cluster with that cluster's top features; avoids the
    25x time cost of fully-independent per-asset models).
  - 0.3-0.6 => mixed; lean toward shared model + cluster-specific feature subsets.

## Per-asset signal strength (wyy_share)
- High-share assets deserve disproportionate model capacity.
- See m3_asset_overview.csv (compare to strategy_perasset/model_meta.json
  per_asset_cv: asset_12/8 strong, asset_10 weak).
"""
    ec.save_summary("module3", md, out / "module3_summary.md")
    print(f"\nJaccard mean (top-20 features across assets) = {jaccard_mean:.3f}", flush=True)
    ec.list_outputs(out)


if __name__ == "__main__":
    main()

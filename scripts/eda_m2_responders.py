"""EDA Module 2 — Responder structure (the "reopen responder?" verdict).

Runs first because it's the cheapest (47 columns) and answers the highest-risk
binary decision: is the 47-responder covariance a *stable global property*
(reopen responder as a training-time auxiliary signal) or a *time-drifting
train-distribution artifact* (keep it frozen, same death mechanism as cs/rolling)?

Outputs to out/eda/:
  m2_responder_corr_heatmap.png      47x47 corr, ordered by horizon group
  m2_responder_pca_scree.png         explained variance + cumulative
  m2_responder_pca_variance.csv
  m2_horizon_group_validation.csv    group -> in-group mean|corr| vs global ratio
  m2_responder_target_corr.csv       47 rows, weighted corr with target, sorted
  m2_cov_consistency.csv             fro / corr_upper / procrustes (the verdict)
  m2_cov_segments_heatmap.png        early/mid/late cov side by side
  module2_summary.md                 decision numbers written out

Usage:
  python scripts/eda_m2_responders.py --data-root data --sample-rows 2000000 --out out/eda
  # cloud full-precision:
  python scripts/eda_m2_responders.py --data-root data --sample-rows 0 --out out/eda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

import eda_common as ec


# Horizon groups parsed from missing_report.csv (subsample of 1M rows, but the
# grouping itself is structural — same missing-count => same future-window
# boundary). Re-derived on full data below; this is the a-priori grouping.
HORIZON_GROUPS_BY_MISSING = {
    504: [30, 46],
    484: [6, 13, 20, 27],
    204: [29, 45],
    184: [5, 12, 19, 26],
    104: [28, 44],
    84: [4, 11, 18, 25],
    24: [37, 42],
    9: [36, 41],
    4: [35, 40],
    2: [1],
    0: [2, 3, 7, 8, 9, 10, 14, 15, 16, 17, 21, 22, 23, 24, 31, 32, 33, 34, 38, 39, 43],
}
# responder_00 has missing=99 (its own singleton); fold into the 0-group listing below.


def responder_index_map(responder_cols):
    """Map responder_00..46 -> index in cols list. Returns dict {int -> idx}."""
    out = {}
    for idx, c in enumerate(responder_cols):
        # c like 'responder_03'
        n = int(c.split("_")[1])
        out[n] = idx
    return out


def order_by_horizon(responder_cols):
    """Order responder columns so same-horizon-group members are adjacent
    (makes the 47x47 heatmap reveal block structure)."""
    idx_map = responder_index_map(responder_cols)
    # Build group assignment: missing count -> responder number.
    num_to_group = {}
    for miss, nums in HORIZON_GROUPS_BY_MISSING.items():
        for n in nums:
            num_to_group[n] = miss
    # Sort by (group, responder_number) so groups cluster.
    def sort_key(c):
        n = int(c.split("_")[1])
        return (num_to_group.get(n, 10_000), n)
    ordered = sorted(responder_cols, key=sort_key)
    return ordered


def horizon_group_validation(corr, responder_cols):
    """For each horizon group with >1 member, compare in-group mean|corr|
    (off-diagonal) to the overall mean|corr|. Ratio >> 1 => grouping is real."""
    idx_map = responder_index_map(responder_cols)
    abs_corr = np.abs(corr)
    n = corr.shape[0]
    # overall mean of off-diagonal |corr|
    off = abs_corr[~np.eye(n, dtype=bool)]
    overall = float(off.mean())
    rows = []
    for miss, nums in HORIZON_GROUPS_BY_MISSING.items():
        if len(nums) < 2:
            continue
        idxs = [idx_map[m] for m in nums if m in idx_map]
        if len(idxs) < 2:
            continue
        sub = abs_corr[np.ix_(idxs, idxs)]
        in_group = sub[~np.eye(len(idxs), dtype=bool)].mean()
        members = ",".join(f"responder_{m:02d}" for m in nums)
        rows.append({
            "missing_count": miss,
            "members": members,
            "n_members": len(nums),
            "in_group_mean_abs_corr": float(in_group),
            "overall_mean_abs_corr": overall,
            "ratio": float(in_group / overall) if overall > 0 else float("nan"),
        })
    return pl.DataFrame(rows), overall


def cov_consistency(R_early, R_mid, R_late, n_pc=5):
    """The core verdict: is the 47x47 responder covariance stable across time?

    Metrics:
      fro_ab     = ||Sigma_a - Sigma_b||_F / ||Sigma_a||_F   (relative Frobenius)
      corr_upper = corr(vech(Sigma_early), vech(Sigma_late)) (upper-triangle vector corr)
      procrustes = mean subspace similarity of top-n_pc loadings across segments
    """
    Sig = {}
    for name, R in [("early", R_early), ("mid", R_mid), ("late", R_late)]:
        Sig[name] = ec.cov_matrix(R)

    def fro(a, b):
        return float(np.linalg.norm(Sig[a] - Sig[b]) / max(np.linalg.norm(Sig[a]), 1e-12))

    def vech(M):
        iu = np.triu_indices_from(M, k=1)
        return M[iu]

    def corr_upper(a, b):
        v1, v2 = vech(Sig[a]), vech(Sig[b])
        if v1.std() < 1e-12 or v2.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(v1, v2)[0, 1])

    def top_loadings(M):
        # PCA via SVD on centered data is already done; here we eigendecompose the
        # covariance and return top-n_pc unit eigenvectors as columns.
        w, V = np.linalg.eigh(M)
        order = np.argsort(w)[::-1][:n_pc]
        return V[:, order]

    def procrustes(a, b):
        A, B = top_loadings(Sig[a]), top_loadings(Sig[b])
        # Mean cosine of principal angles between subspaces = trace(A^T B B^T A)/n_pc
        s = np.linalg.svd(A.T @ B, compute_uv=False)
        return float(np.sum(s[:n_pc]) / n_pc)  # in [0,1], 1=identical subspace

    return {
        "fro_early_mid": fro("early", "mid"),
        "fro_mid_late": fro("mid", "late"),
        "fro_early_late": fro("early", "late"),
        "corr_upper_early_late": corr_upper("early", "late"),
        "corr_upper_early_mid": corr_upper("early", "mid"),
        "corr_upper_mid_late": corr_upper("mid", "late"),
        "procrustes_early_mid": procrustes("early", "mid"),
        "procrustes_early_late": procrustes("early", "late"),
        "procrustes_mid_late": procrustes("mid", "late"),
    }


def segment_sample(lf, t_min, t_max, n_rows, n_assets=15, seed=42, cols=None):
    """Sample within a time_id range, keeping cross-sectional structure.
    If ``cols`` is given, select only those columns before collect (saves a lot
    of memory vs collecting all 375 columns then subsetting)."""
    sub = ec.filter_time_range(lf, t_min, t_max)
    if cols is not None:
        sub = sub.select(["time_id"] + cols)
    return ec.sample_by_time(sub, n_rows, n_assets=n_assets, seed=seed)


def main():
    ap = argparse.ArgumentParser(description="EDA module 2: responder structure")
    ec.add_common_args(ap)
    ap.add_argument("--seg-rows", type=int, default=1_500_000,
                    help="Sample rows per early/mid/late segment (0 = full segment).")
    args = ap.parse_args()

    out = ec.ensure_out_dir(args.out)
    load_train, _ = ec.load_train_lazy(args.data_root)
    lf = load_train(args.data_root)
    responder_cols = ec.get_responder_columns(lf)
    print(f"responders: {len(responder_cols)}", flush=True)

    # ---- 1. 47x47 responder correlation ----
    # 47x47 corr only needs ~500k rows (Pearson SE ~0.0014); cap to keep memory
    # low. The covariance-consistency verdict uses --seg-rows separately below.
    # Select only needed cols before collect so full mode (--sample-rows 0) also
    # stays cheap (50 cols, not 375).
    corr_rows = min(args.sample_rows, 500_000) if args.sample_rows > 0 else 0
    lf_resp = lf.select(["time_id"] + responder_cols + ["target", "weight"])
    sample = ec.sample_by_time(lf_resp, corr_rows, seed=args.seed)
    print(f"sample rows for corr/PCA: {sample.height} (capped at 500k)", flush=True)
    sample_clean = ec.drop_any_nan_rows(sample, responder_cols)
    print(f"after NaN drop: {sample_clean.height}", flush=True)
    corr_n = sample_clean.height
    R = sample_clean.select(responder_cols).to_numpy().astype(np.float32)
    del sample, sample_clean
    corr = ec.corr_matrix(R, assert_shape=(len(responder_cols), len(responder_cols)))

    ordered = order_by_horizon(responder_cols)
    order_idx = [responder_cols.index(c) for c in ordered]
    corr_ordered = corr[np.ix_(order_idx, order_idx)]
    ec.save_corr_heatmap(corr_ordered, ordered,
                         out / "m2_responder_corr_heatmap.png",
                         title="Responder correlation (ordered by horizon group)")

    # ---- 2. PCA on centered R ----
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(15, len(responder_cols)))
    pca.fit(R)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pca_df = pl.DataFrame({
        "pc": [f"pc{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": evr.tolist(),
        "cumulative": cum.tolist(),
    })
    ec.save_csv(pca_df, out / "m2_responder_pca_variance.csv")
    # scree
    series = {"explained variance ratio": (np.arange(1, len(evr) + 1), evr),
              "cumulative": (np.arange(1, len(evr) + 1), cum)}
    ec.save_line(series, out / "m2_responder_pca_scree.png",
                 xlabel="principal component", ylabel="ratio", title="Responder PCA scree")

    # ---- 3. horizon group validation ----
    hg_df, overall_abs = horizon_group_validation(corr, responder_cols)
    ec.save_csv(hg_df, out / "m2_horizon_group_validation.csv",
                note=f"overall mean|corr|={overall_abs:.4f}")

    # ---- 4. responder-target weighted correlation (full-precision streaming) ----
    rc = ec.stream_weighted_corr(lf, responder_cols, target="target", weight="weight")
    rc_rows = sorted(((c, v) for c, v in rc.items()), key=lambda kv: abs(kv[1]), reverse=True)
    rc_df = pl.DataFrame({
        "responder": [c for c, _ in rc_rows],
        "weighted_corr_target": [v for _, v in rc_rows],
        "abs_corr": [abs(v) for _, v in rc_rows],
    })
    ec.save_csv(rc_df, out / "m2_responder_target_corr.csv")

    # ---- 5. early/mid/late covariance consistency (the verdict) ----
    seg_rows = args.seg_rows if args.seg_rows > 0 else 0  # 0 => full segment
    # time boundaries: train partitions align to 100k time_id blocks.
    # Collect only responder cols (not all 375) to bound memory.
    R_early = segment_sample(lf, 0, 300_000, seg_rows, seed=args.seed, cols=responder_cols)
    R_early = ec.drop_any_nan_rows(R_early, responder_cols).to_numpy().astype(np.float32)
    R_mid = segment_sample(lf, 300_000, 600_000, seg_rows, seed=args.seed, cols=responder_cols)
    R_mid = ec.drop_any_nan_rows(R_mid, responder_cols).to_numpy().astype(np.float32)
    R_late = segment_sample(lf, 600_000, 888_480, seg_rows, seed=args.seed, cols=responder_cols)
    R_late = ec.drop_any_nan_rows(R_late, responder_cols).to_numpy().astype(np.float32)
    for name, fr_rows in [("early", R_early.shape[0]), ("mid", R_mid.shape[0]), ("late", R_late.shape[0])]:
        print(f"segment {name}: {fr_rows} rows", flush=True)

    cons = cov_consistency(R_early, R_mid, R_late, n_pc=5)
    cons_df = pl.DataFrame([cons])
    ec.save_csv(cons_df, out / "m2_cov_consistency.csv")

    # side-by-side covariance heatmaps (normalized to correlation scale for color)
    Sig_e = ec.cov_matrix(R_early); Sig_m = ec.cov_matrix(R_mid); Sig_l = ec.cov_matrix(R_late)
    def to_corr(S):
        d = np.sqrt(np.diag(S))
        d[d < 1e-12] = 1e-12
        return S / np.outer(d, d)
    if ec._HAS_MPL:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, S, t in zip(axes, [to_corr(Sig_e), to_corr(Sig_m), to_corr(Sig_l)],
                            ["early", "mid", "late"]):
            im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_title(f"responder corr {t}")
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
        fig.savefig(str(out / "m2_cov_segments_heatmap.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out / 'm2_cov_segments_heatmap.png'}", flush=True)

    # ---- summary ----
    max_abs_target = max((abs(v) for v in rc.values() if np.isfinite(v)), default=0.0)
    top5_cum = float(cum[4]) if len(cum) >= 5 else float(cum[-1])
    need20_pc = int(np.searchsorted(cum, 0.80) + 1) if len(cum) else 99
    fro_max = max(cons["fro_early_mid"], cons["fro_mid_late"], cons["fro_early_late"])
    proc_min = min(cons["procrustes_early_mid"], cons["procrustes_mid_late"], cons["procrustes_early_late"])
    corr_up = cons["corr_upper_early_late"]
    proc = cons["procrustes_early_late"]

    # decision — procrustes (top-PC subspace agreement) and fro (Frobenius norm
    # distance) are the robust metrics. corr_upper (vector corr of the 1081
    # upper-triangle entries) is NOT used as a gate: it is dominated by the many
    # near-zero small covariances whose sign flips with noise, so it can go
    # negative even when the principal structure is perfectly stable (observed:
    # procrustes 0.9999 but corr_upper -0.33). corr_upper is reported as a
    # reference only.
    if proc_min > 0.85 and fro_max < 0.15 and max_abs_target > 0.3:
        verdict = (f"REOPEN responder (top-PC subspace stable across early/mid/late "
                   f"[min procrustes {proc_min:.4f}] and Frobenius distance small "
                   f"[max fro {fro_max:.4f}]; covariance is a stable global property; "
                   f"build PCA/GMM structure features). NOTE corr_upper={corr_up:.3f} is "
                   "a noisy reference, not the gate.")
    elif proc_min < 0.70 or fro_max > 0.30:
        verdict = ("KEEP FROZEN (top-PC subspace drifts [min procrustes "
                   f"{proc_min:.4f}] or Frobenius distance large [max fro {fro_max:.4f}] "
                   "-> same death mechanism as cs/rolling)")
    else:
        verdict = (f"GRAY ZONE (procrustes {proc_min:.3f}, fro {fro_max:.3f}): "
                   "bootstrap-check; if subspace stable above 0.85 try single-partition "
                   "AV-CV cautiously, else keep frozen")

    if need20_pc > 20:
        verdict += " | PCA note: needs >20 PCs for 80% var => responders near-diagonal, little structure to exploit => keep frozen regardless."

    md = f"""# EDA Module 2 — Responder structure

## Verdict
**{verdict}**

## Key numbers
- 47x47 responder correlation computed on {corr_n:,} rows.
- Responder PCA: top-5 PCs explain **{top5_cum:.1%}** of variance; need **{need20_pc}** PCs for 80%.
- Horizon-group validation: in-group mean|corr| / overall mean|corr| ratios (>>1 means grouping real):
{hg_df.to_pandas().to_markdown(index=False) if hasattr(hg_df,'to_pandas') else hg_df}
- Max |weighted corr(responder, target)| = **{max_abs_target:.4f}**
- Top 8 responder-target weighted correlations:
{rc_df.head(8).to_pandas().to_markdown(index=False) if hasattr(rc_df,'to_pandas') else rc_df.head(8)}

## Covariance stability across early/mid/late (the core verdict metric)
{cons_df.to_pandas().to_markdown(index=False) if hasattr(cons_df,'to_pandas') else cons_df}

- procrustes_* = mean subspace similarity of top-5 PC loadings (1.0 = identical). **PRIMARY gate** — robust.
- fro_* = relative Frobenius distance (0 = identical, large = drifted). **PRIMARY gate**.
- corr_upper_* = vector corr of the 1081 upper-triangle entries. REFERENCE ONLY —
  dominated by many near-zero small covariances whose sign flips with noise, so it
  can go negative even when the principal structure is perfectly stable.

## Decision rule applied
- REOPEN if min(procrustes)>0.85 AND max(fro)<0.15 AND max|corr(responder,target)|>0.3
- KEEP FROZEN if min(procrustes)<0.70 OR max(fro)>0.30
- GRAY ZONE otherwise
- Override: if >20 PCs needed for 80% variance => responders near-diagonal => keep frozen.
- min(procrustes across pairs) = {proc_min:.4f}; max(fro across pairs) = {fro_max:.4f}

## Caveat
This run used sample_rows={args.sample_rows} (0 = full). For the verdict metric,
a cloud full-precision rerun (sample-rows 0) is recommended to confirm.
"""
    ec.save_summary("module2", md, out / "module2_summary.md")
    print(f"\nVERDICT: {verdict}", flush=True)
    ec.list_outputs(out)


if __name__ == "__main__":
    main()

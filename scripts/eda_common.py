"""Shared infrastructure for the deep-EDA scripts.

Design goals:
- polars lazy + numpy only (no pandas on the EDA side).
- Two execution modes via ``--sample-rows``:
    * >0 : stratified-by-time_id sample (keep all ~15 assets per sampled time_id).
    * 0  : full collect (for 64G+ cloud runs).
- Streaming single-pass sufficient statistics (weighted correlations, column
  moments) are always full-precision regardless of --sample-rows, because they
  accumulate per-partition and never materialize the full N x K matrix.
- Plotting degrades gracefully: if matplotlib/seaborn are missing (e.g. a
  headless cloud box), PNGs are skipped with a warning and CSVs still land.

The one OOM trap to avoid: ``np.corrcoef(X)`` defaults to ``rowvar=True`` which
builds an N x N matrix (N = rows). Always pass ``rowvar=False`` and assert the
output is (K, K).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

# Plotting is optional. Import lazily so a missing lib never blocks analysis.
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover - environment dependent
    plt = None
    _HAS_MPL = False

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:  # pragma: no cover
    sns = None
    _HAS_SNS = False


# ---------------------------------------------------------------------------
# Column enumeration (delegates to src.features so the source of truth is one place)
# ---------------------------------------------------------------------------

def get_feature_columns(lf):
    return [c for c in lf.collect_schema().names() if c.startswith("feature_")]


def get_responder_columns(lf):
    return [c for c in lf.collect_schema().names() if c.startswith("responder_")]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def total_time_ids(lf):
    """Distinct time_id count via lazy collect (cheap, one column)."""
    return lf.select(pl.col("time_id").n_unique()).collect().item()


def sample_by_time(lf, n_rows, n_assets=15, seed=42):
    """Stratified-by-time_id sample keeping every asset in each chosen time_id.

    Returns a materialized pl.DataFrame. ``n_rows`` is a target row count; the
    actual count is ~ n_rows rounded to a whole time_id slice (each slice has
    ~n_assets rows). With n_rows<=0 the full frame is collected (cloud mode).
    """
    if n_rows is not None and n_rows > 0:
        n_time = total_time_ids(lf)
        target_time_ids = max(1, n_rows // max(1, n_assets))
        if target_time_ids >= n_time:
            return lf.collect()
        # Deterministic stride: pick every k-th time_id, starting at seed % k.
        k = max(1, round(n_time / target_time_ids))
        offset = seed % k
        lf = lf.filter((pl.col("time_id") - offset) % k == 0)
        return lf.collect()
    return lf.collect()


def filter_time_range(lf, t_min, t_max):
    """Lazy frame restricted to a time_id range (half-open [t_min, t_max))."""
    return lf.filter((pl.col("time_id") >= t_min) & (pl.col("time_id") < t_max))


# ---------------------------------------------------------------------------
# Streaming single-pass sufficient statistics (full-precision, near-zero memory)
# ---------------------------------------------------------------------------

def stream_weighted_corr(lf, cols, target="target", weight="weight"):
    """Weighted Pearson correlation of each col in ``cols`` with ``target``.

    Accumulates the six sufficient statistics over a single lazy scan:
      Sw, Swx, Swy, Swxy, Swx2, Swy2
    then corr_w = (Swxy - Swx*Swy/Sw) / sqrt((Swx2-Swx^2/Sw)*(Swy2-Swy^2/Sw)).

    NaNs in any of (col, target, weight) are dropped pairwise per column.
    Returns dict[col -> weighted corr].
    """
    out = {}
    for c in cols:
        # Drop rows where any of c/target/weight is NaN/null for this column.
        good = (
            pl.col(c).is_not_nan()
            & pl.col(c).is_not_null()
            & pl.col(target).is_not_nan()
            & pl.col(target).is_not_null()
            & pl.col(weight).is_not_nan()
            & pl.col(weight).is_not_null()
        )
        x = pl.col(c).cast(pl.Float64)
        y = pl.col(target).cast(pl.Float64)
        w = pl.col(weight).cast(pl.Float64)
        stats = lf.filter(good).select(
            w.sum().alias("Sw"),
            (w * x).sum().alias("Swx"),
            (w * y).sum().alias("Swy"),
            (w * x * y).sum().alias("Swxy"),
            (w * x * x).sum().alias("Swx2"),
            (w * y * y).sum().alias("Swy2"),
        ).collect()
        Sw = stats["Sw"].item()
        if Sw is None or Sw <= 0:
            out[c] = float("nan")
            continue
        Swx = stats["Swx"].item(); Swy = stats["Swy"].item()
        Swxy = stats["Swxy"].item()
        Swx2 = stats["Swx2"].item(); Swy2 = stats["Swy2"].item()
        cov = Swxy - Swx * Swy / Sw
        vx = Swx2 - Swx * Swx / Sw
        vy = Swy2 - Swy * Swy / Sw
        denom = np.sqrt(vx * vy)
        out[c] = float(cov / denom) if denom > 0 else float("nan")
    return out


def stream_col_moments(lf, cols, chunk=64):
    """Per-column distribution moments over a full lazy scan (streaming).

    Returns a pl.DataFrame[col, mean, std, skew, kurt, q01, q05, q50, q95,
    q99, min, max]. Used to reverse-engineer feature physical type from
    distribution shape (heavy-tailed return-like, right-skew vol-like, etc.).

    Columns are processed in chunks because a single select with hundreds of
    aggregation exprs can blow up polars' expression plan / intermediate memory
    on the 13.2M-row scan (a small-Rust-alloc failure masquerading as OOM).
    """
    stat_names = ["mean", "std", "skew", "kurt", "q01", "q05", "q50", "q95", "q99", "min", "max"]
    index = []
    data = {n: [] for n in stat_names}
    for start in range(0, len(cols), chunk):
        batch = cols[start:start + chunk]
        exprs = []
        for c in batch:
            col = pl.col(c).cast(pl.Float64)
            exprs += [
                col.mean().alias(f"{c}__mean"),
                col.std().alias(f"{c}__std"),
                col.skew().alias(f"{c}__skew"),
                col.kurtosis().alias(f"{c}__kurt"),
                col.quantile(0.01).alias(f"{c}__q01"),
                col.quantile(0.05).alias(f"{c}__q05"),
                col.quantile(0.50).alias(f"{c}__q50"),
                col.quantile(0.95).alias(f"{c}__q95"),
                col.quantile(0.99).alias(f"{c}__q99"),
                col.min().alias(f"{c}__min"),
                col.max().alias(f"{c}__max"),
            ]
        row = lf.select(exprs)
        if isinstance(row, pl.LazyFrame):
            row = row.collect()
        for c in batch:
            index.append(c)
            for n in stat_names:
                data[n].append(row[f"{c}__{n}"].item())
    return pl.DataFrame({"column": index, **data})


def weighted_r2_zero_baseline(lf, target="target", weight="weight", by=None):
    """Sum(w*y^2) overall or per-group = R^2 zero-prediction denominator.

    Proportional to the "available signal ceiling" for a group. With ``by`` set
    (e.g. 'asset_id') returns a per-group frame; otherwise a single float.
    """
    w = pl.col(weight).cast(pl.Float64)
    y = pl.col(target).cast(pl.Float64)
    if by is None:
        return lf.select((w * y * y).sum().alias("wyy_sum")).collect().item()
    return (
        lf.group_by(by)
        .agg(
            (w * y * y).sum().alias("wyy_sum"),
            w.sum().alias("w_sum"),
            y.mean().alias("target_mean"),
            y.std().alias("target_std"),
            pl.len().alias("n"),
        )
        .sort(by)
        .collect()
    )


# ---------------------------------------------------------------------------
# numpy helpers (memory-safe)
# ---------------------------------------------------------------------------

def corr_matrix(X, assert_shape=None):
    """Memory-safe correlation matrix of X (n, k). ALWAYS rowvar=False.

    Default np.corrcoef rowvar=True would build an n x n matrix -> OOM for
    n in the millions. Returns (k, k).
    """
    C = np.corrcoef(X, rowvar=False)
    if assert_shape is not None:
        assert C.shape == assert_shape, f"corr matrix shape {C.shape} != {assert_shape}"
    return C


def cov_matrix(R):
    """Covariance matrix of R (n, k) after centering. Returns (k, k)."""
    Rc = R - R.mean(axis=0, keepdims=True)
    return np.cov(Rc, rowvar=False)


def drop_any_nan_rows(df, cols):
    """Drop rows where any of cols is NaN/null. For responder 47x47 work
    (missing <0.4%, loses <1% of rows)."""
    mask = pl.lit(True)
    for c in cols:
        mask = mask & pl.col(c).is_not_nan() & pl.col(c).is_not_null()
    return df.filter(mask)


# ---------------------------------------------------------------------------
# Persistence (CSV always; PNG/MD with graceful degradation)
# ---------------------------------------------------------------------------

def ensure_out_dir(out_dir):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df, path, note=None):
    """Write a polars frame to CSV. Accepts a 2D numpy array by wrapping it."""
    if isinstance(df, np.ndarray):
        df = pl.DataFrame(df)
    df.write_csv(str(path))
    if note:
        print(f"  wrote {path}  ({note})", flush=True)
    else:
        print(f"  wrote {path}", flush=True)


def save_corr_heatmap(mat, labels, path, title="", cmap="RdBu_r", vmin=-1, vmax=1):
    if not _HAS_MPL:
        print(f"  [skip PNG, no matplotlib] {path}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.18 + 2),
                                    max(5, len(labels) * 0.18 + 2)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    if len(labels) <= 60:
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def save_scatter(x, y, path, xlabel="", ylabel="", title=""):
    if not _HAS_MPL:
        print(f"  [skip PNG, no matplotlib] {path}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=6, alpha=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(str(path), dpi=120); plt.close(fig)
    print(f"  wrote {path}", flush=True)


def save_line(series_dict, path, xlabel="time_id bucket", ylabel="", title=""):
    """series_dict: {label: (x_array, y_array)}."""
    if not _HAS_MPL:
        print(f"  [skip PNG, no matplotlib] {path}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, (x, y) in series_dict.items():
        ax.plot(x, y, label=label, linewidth=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(str(path), dpi=120); plt.close(fig)
    print(f"  wrote {path}", flush=True)


def save_dendrogram(linkage_mat, labels, path, title=""):
    if not _HAS_MPL:
        print(f"  [skip PNG, no matplotlib] {path}", flush=True)
        return
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.12), 5))
    dendrogram(linkage_mat, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=6)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(str(path), dpi=120); plt.close(fig)
    print(f"  wrote {path}", flush=True)


def save_summary(module_name, md_text, path):
    Path(path).write_text(md_text, encoding="utf-8")
    print(f"  wrote {path}", flush=True)


def md_table(df, max_rows=None):
    """Render a polars/numpy frame as a GitHub-markdown table without depending
    on pandas/tabulate (cloud envs often lack tabulate). Falls back to a plain
    string if rendering fails.
    """
    try:
        if isinstance(df, np.ndarray):
            df = pl.DataFrame(df)
        if max_rows is not None and hasattr(df, "head"):
            df = df.head(max_rows)
        if not hasattr(df, "columns"):
            return str(df)
        cols = list(df.columns)
        # polars DataFrame -> list of rows
        rows = df.rows()
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, sep]
        for r in rows:
            cells = []
            for v in r:
                if v is None:
                    cells.append("")
                elif isinstance(v, float):
                    cells.append(f"{v:.6g}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
    except Exception as e:
        return f"(table render failed: {e})\n{df}"


def list_outputs(out_dir):
    """Print the produced files (helps pack up results on a cloud box)."""
    p = Path(out_dir)
    print("\n=== EDA outputs ===", flush=True)
    if not p.exists():
        print(f"  (none in {p})", flush=True)
        return
    for f in sorted(p.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:48s} {size/1024:10.1f} KB", flush=True)


# ---------------------------------------------------------------------------
# common CLI parsing so all eda_mN scripts share --data-root/--sample-rows/--out
# ---------------------------------------------------------------------------

def add_common_args(parser):
    parser.add_argument("--data-root", default="data",
                        help="Data root (contains manifest.json) or the train/ dir directly.")
    parser.add_argument("--sample-rows", type=int, default=2_000_000,
                        help="Target sample rows (stratified by time_id). 0 = full collect (cloud).")
    parser.add_argument("--out", default="out/eda", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_train_lazy(data_root):
    """Local import to avoid forcing src on sys.path at module import."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.dataset import load_train, load_test
    return load_train, load_test

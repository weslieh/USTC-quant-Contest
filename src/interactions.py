"""Pairwise feature interactions (products and ratios).

These operate on the *feature axis* (same sample, two raw features), unlike
the cross-sectional / rolling transforms which operate on the sample and time
axes. Because the computation is purely within-row, it is identical at training
(Polars, full panel) and inference (numpy, one time_id slice) — no cross-sample
or cross-time state is involved, which is the key reason this can work where
cs/rolling failed under distribution drift.

A ratio ``a / b`` uses ``b + eps`` to avoid division by zero and ``sign(b)``
would flip the sign — instead we add a small positive epsilon so the ratio
keeps the numerator's sign, matching the inference mirror exactly.
"""

from __future__ import annotations

import numpy as np
import polars as pl

EPS = 1e-8


def make_pairs(source_cols: list[str]) -> list[tuple[str, str]]:
    """All unique unordered pairs (i, j) with i < j from ``source_cols``."""
    n = len(source_cols)
    return [(source_cols[i], source_cols[j]) for i in range(n) for j in range(i + 1, n)]


def interaction_column_names(pairs: list[tuple[str, str]]) -> list[str]:
    """Deterministic ordered list of interaction column names: for each pair
    ``[i_mul_j, i_div_j]`` in pair order."""
    cols = []
    for a, b in pairs:
        cols.append(f"{a}_mul_{b}")
        cols.append(f"{a}_div_{b}")
    return cols


def build_interaction_features(df, pairs: list[tuple[str, str]]):
    """Add ``{a}_mul_{b}`` and ``{a}_div_{b}`` columns for each pair.

    NaN in a source column is treated as 0 within the product (so mul→0) and
    the ratio uses ``b + eps`` with NaN-as-0 too, keeping train/infer parity.
    Returns ``(df_with_cols, new_col_names)``.
    """
    if not pairs:
        return df, []
    exprs = []
    new_cols = []
    for a, b in pairs:
        ca = pl.when(pl.col(a).is_nan()).then(0.0).otherwise(pl.col(a))
        cb = pl.when(pl.col(b).is_nan()).then(0.0).otherwise(pl.col(b))
        mul_name = f"{a}_mul_{b}"
        div_name = f"{a}_div_{b}"
        exprs.append((ca * cb).alias(mul_name))
        exprs.append((ca / (cb + EPS)).alias(div_name))
        new_cols.append(mul_name)
        new_cols.append(div_name)
    return df.with_columns(exprs), new_cols


def interaction_block_numpy(Xraw: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
    """Inference mirror of ``build_interaction_features``.

    Args:
        Xraw: (n_rows, n_raw) float array of raw features (NaNs allowed).
        pair_idx: (n_pairs, 2) int array of column indices into ``Xraw``.

    Returns:
        (n_rows, 2*n_pairs) float32 array laid out as
        ``[mul_0, div_0, mul_1, div_1, ...]`` — exactly the order of
        ``interaction_column_names``.
    """
    Xraw = np.asarray(Xraw, dtype=np.float64)
    # Treat NaN as 0, mirroring the Polars side.
    X = np.where(np.isfinite(Xraw), Xraw, 0.0)
    n = X.shape[0]
    out = np.empty((n, 2 * pair_idx.shape[0]), dtype=np.float32)
    for k, (i, j) in enumerate(pair_idx):
        a = X[:, i]
        b = X[:, j]
        out[:, 2 * k] = a * b
        out[:, 2 * k + 1] = a / (b + EPS)
    return out


def per_asset_column_names(per_asset_specs):
    """Deterministic column names for per-asset masked features.

    ``per_asset_specs`` is a list of ``(asset_id, feature_name)`` pairs. The
    column for ``(a, f)`` is ``pa_{a}_{f}`` — the value of feature ``f`` on
    rows where ``asset_id == a``, and 0 elsewhere. This is a within-row
    transform (no cross-sample / cross-time state), so it is identical at
    training (Polars) and inference (numpy) and safe under distribution drift
    — same rationale as the interaction features above.
    """
    return [f"pa_{a}_{f}" for a, f in per_asset_specs]


def build_per_asset_features(df, per_asset_specs):
    """Add per-asset masked feature columns (Polars, training side).

    For each ``(asset_id a, feature_name f)`` in ``per_asset_specs`` add a
    column ``pa_{a}_{f}`` equal to ``f`` on rows with ``asset_id == a`` and 0
    elsewhere. NaN in ``f`` becomes 0 (mirrored exactly on the numpy side by
    ``per_asset_block_numpy``). Returns ``(df_with_cols, new_col_names)``.
    """
    if not per_asset_specs:
        return df, []
    exprs = []
    new_cols = []
    for a, f in per_asset_specs:
        col = pl.when(pl.col(f).is_nan()).then(0.0).otherwise(pl.col(f))
        exprs.append(
            pl.when(pl.col("asset_id") == a).then(col).otherwise(0.0).alias(f"pa_{a}_{f}")
        )
        new_cols.append(f"pa_{a}_{f}")
    return df.with_columns(exprs), new_cols


def per_asset_block_numpy(Xraw, asset_ids, pa_spec_idx):
    """Inference mirror of ``build_per_asset_features``.

    Args:
        Xraw: (n_rows, n_raw) float array of raw features (NaNs allowed).
        asset_ids: (n_rows,) int array of asset ids.
        pa_spec_idx: (n_specs, 2) int array; each row is
            ``[asset_id, raw_feature_index]``.

    Returns:
        (n_rows, n_specs) float32 array; column k is ``Xraw[:, f]`` on rows
        where ``asset_ids == a`` (NaN->0), else 0. Order matches
        ``per_asset_column_names``.
    """
    Xraw = np.asarray(Xraw, dtype=np.float64)
    X = np.where(np.isfinite(Xraw), Xraw, 0.0)
    asset_ids = np.asarray(asset_ids)
    n = X.shape[0]
    out = np.zeros((n, pa_spec_idx.shape[0]), dtype=np.float32)
    for k, (aid, fidx) in enumerate(pa_spec_idx):
        mask = asset_ids == aid
        if mask.any():
            out[mask, k] = X[mask, fidx]
    return out

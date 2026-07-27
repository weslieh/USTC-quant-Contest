"""Train/inference feature-parity tests.

Asserts that the cross-sectional and rolling features produced by the
training-side Polars helpers (src.features) are reproduced bit-for-bit by the
inference-side numpy helpers (strategy.main._cross_sectional / _RollingBuffer)
when the same data is fed one time_id slice at a time — exactly how the
Time-Series API delivers test rows.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import build_cross_sectional_features, build_rolling_features
from strategy.main import _cross_sectional, _RollingBuffer


def _make_panel(n_times=50, n_assets=15, n_feat=5, seed=0, with_nan=False):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_times):
        for a in range(n_assets):
            row = {"time_id": t, "asset_id": a}
            for f in range(n_feat):
                v = rng.standard_normal()
                if with_nan and rng.random() < 0.05:
                    v = np.nan
                row[f"feature_{f:03d}"] = float(v)
            rows.append(row)
    return pl.DataFrame(rows)


# ---------------- cross-sectional parity ----------------

@pytest.mark.parametrize("with_nan", [False, True])
def test_cross_sectional_parity(with_nan):
    df = _make_panel(n_times=40, n_assets=15, n_feat=5, seed=1, with_nan=with_nan)
    src = ["feature_000", "feature_002", "feature_004"]
    out, new_cols = build_cross_sectional_features(df, src)

    # Inference sees one time_id at a time; rebuild each slice with numpy.
    for t in range(40):
        sl = out.filter(pl.col("time_id") == t)
        raw = sl.select(src).to_numpy().astype(np.float32)
        got = np.zeros((len(sl), len(src) * 3), dtype=np.float32)
        for k in range(len(src)):
            r, z, dm = _cross_sectional(raw[:, k])
            got[:, 3 * k] = r
            got[:, 3 * k + 1] = z
            got[:, 3 * k + 2] = dm
        exp = sl.select([f"{c}_cs_rank" for c in src] +
                        [f"{c}_cs_z" for c in src] +
                        [f"{c}_cs_dm" for c in src]).to_numpy().astype(np.float32)
        # Reorder exp to per-source [rank,z,dm] to match `got`.
        exp_re = np.zeros_like(got)
        for k in range(len(src)):
            exp_re[:, 3 * k] = exp[:, k]                      # rank
            exp_re[:, 3 * k + 1] = exp[:, len(src) + k]       # z
            exp_re[:, 3 * k + 2] = exp[:, 2 * len(src) + k]   # dm
        assert np.allclose(got, exp_re, atol=1e-5, equal_nan=False), \
            f"cs mismatch at time_id={t}"


def test_cross_sectional_degenerate_single_asset():
    # A time_id with only 1 asset -> neutral outputs (rank 0.5, z 0, dm 0).
    df = pl.DataFrame({
        "time_id": [0, 0, 1],
        "asset_id": [0, 1, 0],
        "feature_000": [1.0, 2.0, 5.0],
    })
    out, _ = build_cross_sectional_features(df, ["feature_000"])
    # time_id 1 has a single asset
    single = out.filter(pl.col("time_id") == 1)
    assert single["feature_000_cs_rank"][0] == pytest.approx(0.5)
    assert single["feature_000_cs_z"][0] == pytest.approx(0.0)
    assert single["feature_000_cs_dm"][0] == pytest.approx(0.0)
    # numpy side
    r, z, dm = _cross_sectional(np.array([5.0], dtype=np.float32))
    assert r[0] == pytest.approx(0.5) and z[0] == 0.0 and dm[0] == 0.0


# ---------------- rolling parity ----------------

def test_rolling_parity():
    n_feat, n_assets, n_times, windows = 3, 6, 30, [5, 10]
    df = _make_panel(n_times=n_times, n_assets=n_assets, n_feat=n_feat, seed=2)
    src = ["feature_000", "feature_001", "feature_002"]
    # build_rolling_features sorts by (asset_id, time_id); tag original row
    # order so we can un-sort the output back to df's (time, asset) order.
    df = df.with_row_index("_orig_idx")
    out = build_rolling_features(df, src, windows=tuple(windows))
    # build_rolling_features sorts by (asset_id, time_id); build expected names.
    roll_cols = []
    for s in src:
        roll_cols.append(f"{s}_lag1")
        for w in windows:
            roll_cols += [f"{s}_rm_{w}", f"{s}_rs_{w}"]

    buf = _RollingBuffer(n_source=len(src), windows=windows)
    # Feed time_ids in ascending order, per asset rows in the slice.
    # Pre-compute per-time_id raw matrices in original (time,asset) order.
    raw_full = df.select(src).to_numpy().astype(np.float32)
    tids = df["time_id"].to_numpy()
    aids = df["asset_id"].to_numpy()

    got = np.full((len(df), len(roll_cols)), np.nan, dtype=np.float32)
    for t in range(n_times):
        mask = tids == t
        idxs = np.where(mask)[0]
        for i in idxs:
            aid = int(aids[i])
            row = raw_full[i]
            got[i] = buf.compute(aid, row)
        for i in idxs:  # push after compute (current excluded)
            buf.push(int(aids[i]), raw_full[i])

    # out is in (asset_id, time_id) order; un-sort to original row order via _orig_idx.
    order = np.argsort(out["_orig_idx"].to_numpy())
    exp = out.select(roll_cols).to_numpy()[order].astype(np.float32)
    # NaN->0 alignment: train Polars shift(1) yields null (->NaN via to_numpy)
    # for the first row per asset; inference yields 0.0. Replace NaN with 0.
    exp = np.nan_to_num(exp, nan=0.0, posinf=0.0, neginf=0.0)
    got = np.nan_to_num(got, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.allclose(got, exp, atol=1e-4), \
        f"rolling mismatch, max diff = {np.max(np.abs(got - exp))}"


def test_rolling_excludes_current_row():
    # With a single asset and one feature, lag1 at t must equal the value at t-1
    # (current row excluded), confirming the shift(1) semantics match.
    vals = np.arange(1, 11, dtype=np.float32)
    df = pl.DataFrame({
        "time_id": np.arange(10, dtype=np.int64),
        "asset_id": np.zeros(10, dtype=np.int64),
        "feature_000": vals,
    })
    out = build_rolling_features(df, ["feature_000"], windows=(3,))
    buf = _RollingBuffer(n_source=1, windows=[3])
    got = np.zeros((10, 3), dtype=np.float32)  # [lag1, rm_3, rs_3]
    for t in range(10):
        got[t] = buf.compute(0, np.array([vals[t]]))
        buf.push(0, np.array([vals[t]]))
    exp = out.select(["feature_000_lag1", "feature_000_rm_3", "feature_000_rs_3"]).to_numpy().astype(np.float32)
    exp = np.nan_to_num(exp, nan=0.0)
    assert np.allclose(got, exp, atol=1e-4)
    # lag1 at t=3 == value at t=2 == 3.0 (current excluded)
    assert got[3, 0] == pytest.approx(3.0)


# ---------------- interaction parity ----------------

def test_interaction_parity():
    """Pairwise mul/div interactions must match between the Polars training
    helper (src.interactions) and the numpy inference mirror
    (strategy.main._interaction_block), fed one time_id at a time."""
    from src.interactions import make_pairs, build_interaction_features, interaction_column_names
    from strategy.main import _interaction_block

    rng = np.random.default_rng(7)
    n_times, n_assets, n_feat = 12, 15, 5
    rows = []
    for t in range(n_times):
        for a in range(n_assets):
            row = {"time_id": t, "asset_id": a}
            for f in range(n_feat):
                v = rng.standard_normal()
                if rng.random() < 0.05:
                    v = np.nan
                if rng.random() < 0.05:
                    v = 0.0
                row[f"feature_{f:03d}"] = float(v)
            rows.append(row)
    df = pl.DataFrame(rows)
    raw_cols = [f"feature_{f:03d}" for f in range(n_feat)]
    pairs = make_pairs(raw_cols[:4])  # 4 cols -> 6 pairs -> 12 interaction cols
    col_names = interaction_column_names(pairs)
    out, new_cols = build_interaction_features(df, pairs)
    assert new_cols == col_names

    # Inference builds features per time_id slice; interaction is within-row so
    # we can verify the whole frame at once (slice-agnostic).
    Xraw = df.select(raw_cols).to_numpy().astype(np.float32)
    pair_idx = np.array([[raw_cols.index(a), raw_cols.index(b)] for a, b in pairs], dtype=np.intp)
    got = _interaction_block(Xraw, pair_idx)
    exp = out.select(col_names).to_numpy().astype(np.float32)
    assert got.shape == exp.shape, (got.shape, exp.shape)
    assert np.allclose(got, exp, atol=1e-5, equal_nan=False), \
        f"interaction mismatch, max diff = {np.max(np.abs(got - exp))}"


# ---------------- per-asset masked feature parity ----------------

def test_per_asset_parity():
    """Per-asset masked features (feature value on that asset's rows, 0 else)
    must match between the Polars training helper (src.interactions) and the
    numpy inference mirror (strategy.main._per_asset_block), fed one time_id
    at a time. Includes NaN handling and the asset-mismatch-must-be-zero rule."""
    from src.interactions import build_per_asset_features, per_asset_column_names
    from strategy.main import _per_asset_block

    rng = np.random.default_rng(11)
    n_times, n_assets, n_feat = 12, 15, 5
    rows = []
    for t in range(n_times):
        for a in range(n_assets):
            row = {"time_id": t, "asset_id": a}
            for f in range(n_feat):
                v = rng.standard_normal()
                if rng.random() < 0.05:
                    v = np.nan
                row[f"feature_{f:03d}"] = float(v)
            rows.append(row)
    df = pl.DataFrame(rows)
    raw_cols = [f"feature_{f:03d}" for f in range(n_feat)]
    # Specs: a few (asset_id, feature) pairs, including an asset whose feature
    # has NaN on some rows, and a feature reused across assets.
    specs = [(0, "feature_000"), (0, "feature_003"), (3, "feature_002"),
             (3, "feature_000"), (14, "feature_004")]
    col_names = per_asset_column_names(specs)
    out, new_cols = build_per_asset_features(df, specs)
    assert new_cols == col_names

    # Inference: verify per time_id slice (the real inference path) AND whole
    # frame (within-row so slice-agnostic).
    Xraw = df.select(raw_cols).to_numpy().astype(np.float32)
    asset_ids = df["asset_id"].to_numpy()
    pa_spec_idx = np.array([[a, raw_cols.index(f)] for a, f in specs], dtype=np.intp)

    # whole frame
    got_full = _per_asset_block(Xraw, asset_ids, pa_spec_idx)
    exp = out.select(col_names).to_numpy().astype(np.float32)
    assert got_full.shape == exp.shape, (got_full.shape, exp.shape)
    assert np.allclose(got_full, exp, atol=1e-5, equal_nan=False), \
        f"per-asset mismatch (full), max diff = {np.max(np.abs(got_full - exp))}"

    # per time_id slice (mirrors real inference: one time_id at a time)
    for t in range(n_times):
        mask = df["time_id"].to_numpy() == t
        got_slice = _per_asset_block(Xraw[mask], asset_ids[mask], pa_spec_idx)
        exp_slice = exp[mask]
        assert got_slice.shape == exp_slice.shape
        assert np.allclose(got_slice, exp_slice, atol=1e-5, equal_nan=False), \
            f"per-asset mismatch at time_id={t}"

    # Rule: asset != spec asset => column must be exactly 0.
    for k, (aid, _f) in enumerate(specs):
        col = got_full[:, k]
        other = asset_ids != aid
        assert np.all(col[other] == 0.0), f"spec {k}: nonzero on non-{aid} rows"

    # Column order matches specs order.
    assert col_names == [f"pa_{a}_{f}" for a, f in specs]


# ---------------- target rank transform parity ----------------

def test_target_rank_transform():
    """Per-time_id rank + global inverse-CDF LUT must round-trip, and the
    numpy inference mirror (_inverse_cdf_map) must match the src helper."""
    from src.target_transform import target_rank_per_time, build_inverse_cdf_lut, inverse_cdf_map
    from strategy.main import _inverse_cdf_map

    rng = np.random.default_rng(0)
    time_ids = np.repeat(np.arange(5), 15).astype(np.int64)
    target = rng.standard_normal(75).astype(np.float32)
    target[0] = target[1]  # inject a tie

    # rank per time_id == pandas groupby rank pct average
    got = target_rank_per_time(time_ids, target)
    exp = pd.Series(target).groupby(pd.Series(time_ids)).rank(
        method="average", pct=True).to_numpy()
    assert np.allclose(got, exp, atol=1e-6)
    assert got.min() > 0 and got.max() <= 1.0

    # LUT round-trips quantiles
    y = rng.standard_normal(10000)
    lut = build_inverse_cdf_lut(y, n_points=101)
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    mapped = inverse_cdf_map(qs, lut)
    assert np.allclose(mapped, np.quantile(y, qs), atol=1e-3)

    # inference mirror matches src
    assert np.allclose(_inverse_cdf_map(qs, lut), mapped, atol=1e-5)



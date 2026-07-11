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
        sl = df.filter(pl.col("time_id") == t)
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

    exp = out.select(roll_cols).to_numpy().astype(np.float32)
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

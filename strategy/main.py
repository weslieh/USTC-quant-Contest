from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

EPS = 1e-8


def _cross_sectional(values: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-time_id rank fraction / zscore / demean, mirroring
    src.features.build_cross_sectional_features.

    ``values`` is a 1-D array holding one feature column across all assets in
    a single time_id slice (the order the API delivered them). NaNs are treated
    as 0, matching src.features.build_cross_sectional_features. n<2 yields
    neutral outputs (rank 0.5, z 0, dm 0).
    """
    n = values.shape[0]
    finite = np.isfinite(values)
    out_rank = np.full(n, 0.5, dtype=np.float32)
    out_z = np.zeros(n, dtype=np.float32)
    out_dm = np.zeros(n, dtype=np.float32)
    if n < 2 or not finite.any():
        return out_rank, out_z, out_dm

    # Mirror training: treat NaN as 0 within the cross-sectional computation.
    vals = np.where(finite, values, 0.0).astype(np.float64)
    mean = vals.mean()
    std = vals.std(ddof=1) if n > 1 else 0.0

    # average rank (ties share the mean of their positions) via pandas.
    rank_frac = (pd.Series(vals).rank(method="average").to_numpy() - 1.0) / (n - 1.0)

    z = np.where(std > eps, (vals - mean) / (std + eps), 0.0)
    dm = vals - mean
    return rank_frac.astype(np.float32), z.astype(np.float32), dm.astype(np.float32)


class _RollingBuffer:
    """Per-asset rolling history mirroring src.features.build_rolling_features
    (shift(1) then rolling over history, current row excluded).

    Each asset owns a fixed (max_w, n_source) numpy buffer (oldest at index 0)
    plus a fill count. compute/push are vectorized over source columns.
    """

    def __init__(self, n_source, windows, n_assets_hint=16):
        self.n_source = n_source
        self.windows = list(windows)
        self.max_w = max(self.windows) if self.windows else 1
        self.n_assets_hint = n_assets_hint
        self._buf = {}   # asset_id -> (np.ndarray[max_w, n_source], fill_count)

    def _get(self, asset_id):
        b = self._buf.get(asset_id)
        if b is None:
            arr = np.zeros((self.max_w, self.n_source), dtype=np.float64)
            b = [arr, 0]
            self._buf[asset_id] = b
        return b

    def compute(self, asset_id, current_row):
        """Return flat vector per source: [lag1, rm_w1, rs_w1, rm_w2, rs_w2, ...]."""
        arr, fill = self._get(asset_id)
        out = np.empty(self.n_source * (1 + 2 * len(self.windows)), dtype=np.float32)
        if fill == 0:
            out[:] = 0.0
            return out
        # history is arr[0:fill], oldest..newest; lag1 = newest.
        hist = arr[:fill]
        newest = hist[-1]                      # (n_source,)
        o = 0
        for k in range(self.n_source):
            out[o] = newest[k]
            o += 1
            col = hist[:, k]
            for w in self.windows:
                window = col[-w:]
                out[o] = window.mean()
                o += 1
                out[o] = window.std(ddof=1) if window.size > 1 else 0.0
                o += 1
        return out

    def push(self, asset_id, current_row):
        arr, fill = self._get(asset_id)
        row = np.asarray(current_row, dtype=np.float64)
        if fill < self.max_w:
            arr[fill] = row
            self._buf[asset_id][1] = fill + 1
        else:
            arr[:-1] = arr[1:]
            arr[-1] = row


class Model:
    """Inference model for the Time-Series API.

    Loads an ensemble of fold boosters plus the engineered-feature spec saved
    by train.py. ``predict(test)`` receives one time_id (columns:
    row_id,time_id,asset_id,feature_*) in ascending time order, builds raw +
    cross-sectional + rolling features to match training, averages the fold
    boosters, and clips to a finite range.
    """

    def __init__(self):
        here = Path(__file__).resolve().parent
        meta_path = here / "model_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self.raw_feature_columns = list(meta["raw_feature_columns"])
        self.cs_source_columns = list(meta.get("cs_source_columns", []))
        self.cs_feature_columns = list(meta.get("cs_feature_columns", []))
        self.rolling_source_columns = list(meta.get("rolling_source_columns", []))
        self.rolling_feature_columns = list(meta.get("rolling_feature_columns", []))
        self.rolling_windows = list(meta.get("rolling_windows", []))
        self.feature_columns = list(meta["feature_columns"])  # full ordered list
        self.target_std = float(meta.get("target_std") or 1.0)

        self.boosters = []
        for fname in meta.get("fold_files", []):
            self.boosters.append(lgb.Booster(model_file=str(here / fname)))
        if not self.boosters:  # legacy single model
            self.boosters.append(lgb.Booster(model_file=str(here / "model.txt")))

        # Precompute raw column positions so we can slice the numpy matrix once
        # per time_id instead of a pandas .loc label lookup over 323 columns.
        self._cs_src_idx = np.asarray(
            [self.raw_feature_columns.index(c) for c in self.cs_source_columns
             if c in self.raw_feature_columns], dtype=np.intp
        )
        self._roll_src_idx = np.asarray(
            [self.raw_feature_columns.index(c) for c in self.rolling_source_columns
             if c in self.raw_feature_columns], dtype=np.intp
        )

        self.rolling = _RollingBuffer(len(self._roll_src_idx), self.rolling_windows)
        self.last_time_id: int | None = None

    def _build_features(self, test: pd.DataFrame) -> np.ndarray:
        asset_ids = test["asset_id"].to_numpy()
        # One numpy conversion for all raw columns, then position-index.
        Xraw = test.to_numpy(dtype=np.float32, copy=True)
        # test columns include row_id/time_id/asset_id first; map raw columns to
        # their positions in `test.columns`.
        col_pos = test.columns.get_indexer(self.raw_feature_columns)
        Xraw = Xraw[:, col_pos]
        np.nan_to_num(Xraw, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)

        n = Xraw.shape[0]
        feats = [Xraw]

        # Cross-sectional: per time_id slice (the whole `test` is one time_id).
        if self._cs_src_idx.size:
            cs_block = np.zeros((n, self._cs_src_idx.size * 3), dtype=np.float32)
            for k, ci in enumerate(self._cs_src_idx):
                r, z, dm = _cross_sectional(Xraw[:, ci])
                cs_block[:, 3 * k] = r
                cs_block[:, 3 * k + 1] = z
                cs_block[:, 3 * k + 2] = dm
            feats.append(cs_block)

        # Rolling: per-asset history (current row excluded), push after compute.
        if self._roll_src_idx.size and self.rolling_windows:
            n_roll = self._roll_src_idx.size
            block_w = 1 + 2 * len(self.rolling_windows)
            roll_block = np.zeros((n, n_roll * block_w), dtype=np.float32)
            for i in range(n):
                aid = int(asset_ids[i])
                roll_block[i] = self.rolling.compute(aid, Xraw[i, self._roll_src_idx])
            for i in range(n):
                self.rolling.push(int(asset_ids[i]), Xraw[i, self._roll_src_idx])
            feats.append(roll_block)

        return np.concatenate(feats, axis=1)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must strictly increase in Time-Series API order")
        self.last_time_id = time_id

        X = self._build_features(test)
        # Raw block (first len(raw) cols) keeps NaN for LGBM; engineered blocks
        # are already finite (cs/rolling produce 0 on degenerate input). Only
        # sanitize any residual inf across the full matrix; leave NaN in place
        # so LGBM treats raw NaN as missing exactly like in training.
        np.nan_to_num(X, copy=False, nan=np.nan, posinf=0.0, neginf=0.0)

        preds = np.zeros(X.shape[0], dtype=np.float64)
        for booster in self.boosters:
            preds += booster.predict(X)
        preds /= len(self.boosters)

        clip = 3.0 * self.target_std
        preds = np.clip(preds, -clip, clip)
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0
        return preds

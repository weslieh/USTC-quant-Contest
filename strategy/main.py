from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


class Model:
    """Inference model for the Time-Series API.

    Loads a pretrained LightGBM booster + feature column list saved by
    train.py. `predict(test)` receives a pandas DataFrame holding a single
    time_id (columns: row_id, time_id, asset_id, feature_*), in ascending
    time order, and must return a 1-D finite float array of len(test).

    Supports cross-sectional features (rank/zscore/demean per time_id) and
    rolling features (lag1/rolling_mean/rolling_std per asset) computed at
    inference time to match training-time feature engineering.
    """

    # Suffixes that identify derived feature columns.
    _DERIVED_SUFFIXES = ("_csrank", "_csz", "_csdm", "_lag1")

    def __init__(self):
        here = Path(__file__).resolve().parent
        model_path = here / "model.txt"
        meta_path = here / "model_meta.json"

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_columns = list(meta["feature_columns"])
        self.rolling_windows = list(meta.get("rolling_windows", []))
        self.cross_sectional = meta.get("cross_sectional", False)

        self.booster = lgb.Booster(model_file=str(model_path))
        self.last_time_id: int | None = None

        # Determine raw feature columns by stripping derived suffixes.
        self.raw_cols = [
            c for c in self.feature_columns
            if c.startswith("feature_")
            and not any(c.endswith(s) for s in self._DERIVED_SUFFIXES)
            and "_rm_" not in c
            and "_rs_" not in c
        ]
        self.raw_cols.sort(key=lambda c: int(c.split("_")[1]))
        self.n_raw = len(self.raw_cols)

        # Per-asset rolling history: asset_id -> deque of raw feature vectors.
        self.max_window = max(self.rolling_windows) if self.rolling_windows else 0
        self.asset_history: dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Feature engineering (mirrors src/features.py training logic)
    # ------------------------------------------------------------------

    def _build_features(self, test: pd.DataFrame) -> np.ndarray:
        """Build the full feature matrix matching training-time column order."""
        n = len(test)
        # Extract raw features as (n, n_raw) float32 array.
        raw = np.array(
            test.loc[:, self.raw_cols].to_numpy(), dtype=np.float32, copy=True
        )
        # Replace inf with NaN so downstream stats are clean.
        raw[~np.isfinite(raw)] = np.nan

        asset_ids = test["asset_id"].to_numpy()
        feat_dict: dict[str, np.ndarray] = {}

        # Store raw features.
        for j, col in enumerate(self.raw_cols):
            feat_dict[col] = raw[:, j]

        # --- Cross-sectional features (per time_id, across assets) ---
        if self.cross_sectional:
            self._add_cross_sectional(raw, feat_dict)

        # --- Rolling features (per asset, across time) ---
        if self.rolling_windows:
            self._add_rolling(raw, asset_ids, feat_dict)

        # Assemble final matrix in exact training column order.
        X = np.empty((n, len(self.feature_columns)), dtype=np.float32)
        for k, col in enumerate(self.feature_columns):
            arr = feat_dict.get(col)
            if arr is not None:
                X[:, k] = arr
            else:
                X[:, k] = 0.0  # should not happen if meta is consistent
        return X

    def _add_cross_sectional(
        self, raw: np.ndarray, feat_dict: dict[str, np.ndarray]
    ) -> None:
        """Compute rank, zscore, demean per time_id across assets.

        Mirrors build_cross_sectional_features in src/features.py.
        Uses ddof=1 for std to match Polars default.
        """
        n_valid = np.sum(~np.isnan(raw), axis=0)  # (n_raw,)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(raw, axis=0)  # (n_raw,)
            std = np.nanstd(raw, axis=0, ddof=1)  # (n_raw,)

        demean = raw - mean  # (n, n_raw) via broadcasting

        # zscore: (x - mean) / std, 0 where std == 0
        safe_std = np.where(std > 0, std, 1.0)
        zscore = (raw - mean) / safe_std  # (n, n_raw)
        zscore[:, std <= 0] = 0.0

        # Fractional rank in [0, 1] using "average" method (matches Polars).
        ranks = (
            pd.DataFrame(raw, columns=self.raw_cols)
            .rank(method="average")
            .to_numpy()
        )  # (n, n_raw)
        denom = np.maximum(n_valid - 1, 1)  # (n_raw,)
        rank_frac = (ranks - 1) / denom  # (n, n_raw)
        # Where n_valid <= 1, set rank to 0.5 (matches Polars guard).
        rank_frac[:, n_valid <= 1] = 0.5

        for j, col in enumerate(self.raw_cols):
            feat_dict[f"{col}_csrank"] = rank_frac[:, j].astype(np.float32)
            feat_dict[f"{col}_csz"] = zscore[:, j].astype(np.float32)
            feat_dict[f"{col}_csdm"] = demean[:, j].astype(np.float32)

    def _add_rolling(
        self,
        raw: np.ndarray,
        asset_ids: np.ndarray,
        feat_dict: dict[str, np.ndarray],
    ) -> None:
        """Compute lag1, rolling_mean, rolling_std per asset across time.

        Mirrors build_rolling_features in src/features.py.
        Rolling stats include the current value (min_periods=1).
        lag1 is the previous time_id's value (shift(1)).
        Uses ddof=1 for std to match Polars default.
        """
        n = len(raw)
        lag1 = np.full((n, self.n_raw), np.nan, dtype=np.float32)
        rm = {w: np.full((n, self.n_raw), np.nan, dtype=np.float32) for w in self.rolling_windows}
        rs = {w: np.full((n, self.n_raw), np.nan, dtype=np.float32) for w in self.rolling_windows}

        for i, aid in enumerate(asset_ids):
            aid = int(aid)
            hist = self.asset_history.get(aid)
            hist_list = list(hist) if hist else []

            # lag1: previous value (not current).
            if hist_list:
                lag1[i] = np.asarray(hist_list[-1], dtype=np.float32)

            # Rolling mean/std: include current value, use last w values.
            for w in self.rolling_windows:
                # Window = last (w-1) from history + current value.
                n_take = min(len(hist_list), w - 1)
                if n_take > 0:
                    window = np.vstack([
                        np.asarray(hist_list[-n_take:], dtype=np.float32),
                        raw[i],
                    ])  # (n_take + 1, n_raw)
                else:
                    window = raw[i: i + 1]  # (1, n_raw)

                rm[w][i] = np.nanmean(window, axis=0)
                if len(window) > 1:
                    rs[w][i] = np.nanstd(window, axis=0, ddof=1)
                # else: rs stays NaN (std of 1 value is undefined)

        for j, col in enumerate(self.raw_cols):
            feat_dict[f"{col}_lag1"] = lag1[:, j]
            for w in self.rolling_windows:
                feat_dict[f"{col}_rm_{w}"] = rm[w][:, j]
                feat_dict[f"{col}_rs_{w}"] = rs[w][:, j]

    def _update_history(self, test: pd.DataFrame, raw: np.ndarray) -> None:
        """Update per-asset history with current time_id's raw features."""
        asset_ids = test["asset_id"].to_numpy()
        for i, aid in enumerate(asset_ids):
            aid = int(aid)
            if aid not in self.asset_history:
                self.asset_history[aid] = deque(maxlen=self.max_window + 1)
            self.asset_history[aid].append(raw[i].copy())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must strictly increase in Time-Series API order")
        self.last_time_id = time_id

        X = self._build_features(test)
        # Replace inf with 0 (matching training fill_infinite). Keep NaN for
        # LightGBM to handle natively (matching training behavior).
        inf_mask = np.isinf(X)
        if inf_mask.any():
            X[inf_mask] = 0.0

        preds = self.booster.predict(X)
        preds = np.asarray(preds, dtype=np.float64).reshape(-1)

        # Guarantee finite output (safety net for the eval API).
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0

        # Update history AFTER prediction to avoid future-info leakage.
        raw = np.array(
            test.loc[:, self.raw_cols].to_numpy(), dtype=np.float32, copy=True
        )
        self._update_history(test, raw)

        return preds

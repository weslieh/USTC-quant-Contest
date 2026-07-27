from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

EPS = 1e-8


def _interaction_block(Xraw: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
    """Inference mirror of src.interactions.build_interaction_features.

    Self-contained (no src import) so it works in a stripped submission
    package. Layout: [mul_0, div_0, mul_1, div_1, ...] in pair order.
    Vectorized over pairs (no Python per-pair loop) to stay within the
    per-step time budget across ~214k time_ids.
    """
    X = np.where(np.isfinite(Xraw), Xraw.astype(np.float64), 0.0)
    a = X[:, pair_idx[:, 0]]  # (n, n_pairs)
    b = X[:, pair_idx[:, 1]]  # (n, n_pairs)
    mul = a * b
    div = a / (b + EPS)
    # interleave [mul_k, div_k] per pair -> (n, 2*n_pairs)
    out = np.empty((X.shape[0], 2 * pair_idx.shape[0]), dtype=np.float32)
    out[:, 0::2] = mul
    out[:, 1::2] = div
    return out


def _per_asset_block(Xraw: np.ndarray, asset_ids: np.ndarray, pa_spec_idx: np.ndarray) -> np.ndarray:
    """Inference mirror of src.interactions.build_per_asset_features.

    Self-contained (no src import). For each spec (asset_id a, raw feature idx f)
    the column is Xraw[:, f] on rows where asset_ids == a (NaN->0), else 0.
    Within-row, drift-safe. ``pa_spec_idx`` is (n_specs, 2): [asset_id, feat_idx].
    """
    X = np.where(np.isfinite(Xraw), Xraw.astype(np.float64), 0.0)
    asset_ids = np.asarray(asset_ids)
    out = np.zeros((X.shape[0], pa_spec_idx.shape[0]), dtype=np.float32)
    for k in range(pa_spec_idx.shape[0]):
        aid = pa_spec_idx[k, 0]
        fidx = pa_spec_idx[k, 1]
        mask = asset_ids == aid
        if mask.any():
            out[mask, k] = X[mask, fidx]
    return out


def _neutralize_predictions(preds: np.ndarray, features: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Orthogonalize predictions against features (self-contained mirror of
    src.neutralization.neutralize_predictions). Subtracts ``alpha`` of the
    linear exposure of ``preds`` on ``features`` (with intercept).
    """
    n = preds.shape[0]
    if n < 2 or alpha <= 0.0:
        return preds
    features_clean = np.where(np.isfinite(features), features, 0.0)
    X = np.concatenate([features_clean, np.ones((n, 1))], axis=1)
    try:
        w, _, _, _ = np.linalg.lstsq(X, preds, rcond=None)
        return preds - alpha * (X @ w)
    except np.linalg.LinAlgError:
        return preds


def _inverse_cdf_map(rank_pred: np.ndarray, lut: dict) -> np.ndarray:
    """Map predicted ranks in [0,1] back to target scale via the inverse-CDF
    LUT stored in model_meta.json (self-contained mirror of
    src.target_transform.inverse_cdf_map). Linear interpolation; clamped to
    the LUT value range.
    """
    q = np.asarray(lut["q"], dtype=np.float64)
    v = np.asarray(lut["v"], dtype=np.float64)
    r = np.clip(np.asarray(rank_pred, dtype=np.float64), 0.0, 1.0)
    return np.interp(r, q, v).astype(np.float64)


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
        self.backend = meta.get("backend", "lgb")
        # Target rank transform: map predicted rank -> target scale via LUT.
        self.target_transform = meta.get("target_transform", "none")
        self.target_quantile_lut = meta.get("target_quantile_lut") or None
        # asset_id prepended as the first feature column (categorical for LGB).
        self.asset_as_categorical = bool(meta.get("asset_as_categorical", False))

        self.neutralize_alpha = meta.get("neutralize_alpha", 0.0)
        self.neutralize_features = meta.get("neutralize_features", [])
        self._neutralize_src_idx = np.asarray(
            [self.raw_feature_columns.index(c) for c in self.neutralize_features
             if c in self.raw_feature_columns], dtype=np.intp
        )

        # Interaction feature spec: list of [col_a, col_b] pairs (raw names).
        self.interaction_pairs = [tuple(p) for p in meta.get("interaction_source_columns", [])]
        self._inter_pair_idx = np.asarray(
            [[self.raw_feature_columns.index(a), self.raw_feature_columns.index(b)]
             for a, b in self.interaction_pairs
             if a in self.raw_feature_columns and b in self.raw_feature_columns],
            dtype=np.intp,
        ).reshape(-1, 2)

        # Per-asset masked feature spec: list of [asset_id, feature_name].
        self.per_asset_specs = [tuple(s) for s in meta.get("per_asset_specs", [])]
        self._pa_spec_idx = np.asarray(
            [[int(aid), self.raw_feature_columns.index(fname)]
             for aid, fname in self.per_asset_specs
             if fname in self.raw_feature_columns],
            dtype=np.intp,
        ).reshape(-1, 2) if self.per_asset_specs else np.empty((0, 2), dtype=np.intp)

        self.boosters = []
        if self.backend == "xgb_mt":
            from xgboost import XGBRegressor, XGBClassifier
            self.clf_boosters = []
            self.reg_boosters = []
            for fname in meta.get("fold_files", []):
                clf = XGBClassifier()
                clf.load_model(str(here / f"{fname}_clf.json"))
                self.clf_boosters.append(clf)

                reg = XGBRegressor()
                reg.load_model(str(here / f"{fname}_reg.json"))
                self.reg_boosters.append(reg)
        elif self.backend == "cat_mt":
            from catboost import CatBoostRegressor, CatBoostClassifier
            self.clf_boosters = []
            self.reg_boosters = []
            for fname in meta.get("fold_files", []):
                clf = CatBoostClassifier()
                clf.load_model(str(here / f"{fname}_clf.cbm"))
                self.clf_boosters.append(clf)

                reg = CatBoostRegressor()
                reg.load_model(str(here / f"{fname}_reg.cbm"))
                self.reg_boosters.append(reg)
        elif self.backend == "xgb":
            from xgboost import XGBRegressor
            for fname in meta.get("fold_files", []):
                m = XGBRegressor()
                m.load_model(str(here / fname))
                self.boosters.append(m)
        elif self.backend == "cat":
            from catboost import CatBoostRegressor
            for fname in meta.get("fold_files", []):
                m = CatBoostRegressor()
                m.load_model(str(here / fname))
                self.boosters.append(m)
        else:
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

        # asset_id prepended as column 0 when trained with --asset-as-categorical.
        # Matches train.py's all_feature_cols = ["asset_id"] + raw + engineered.
        if self.asset_as_categorical:
            asset_col = asset_ids.astype(np.float32).reshape(-1, 1)
            feats = [asset_col, Xraw]

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

        # Interactions: within-row pairwise mul/div (feature axis), no state.
        if self._inter_pair_idx.size:
            feats.append(_interaction_block(Xraw, self._inter_pair_idx))

        # Per-asset masked features: within-row, no state (drift-safe).
        if self._pa_spec_idx.size:
            feats.append(_per_asset_block(Xraw, asset_ids, self._pa_spec_idx))

        return np.concatenate(feats, axis=1)

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must strictly increase in Time-Series API order")
        self.last_time_id = time_id

        # Keep a copy of raw features for neutralization later
        Xraw_full = test.to_numpy(dtype=np.float32, copy=True)
        col_pos = test.columns.get_indexer(self.raw_feature_columns)
        Xraw_full = Xraw_full[:, col_pos]

        X = self._build_features(test)
        # Raw block (first len(raw) cols) keeps NaN for LGBM; engineered blocks
        # are already finite (cs/rolling produce 0 on degenerate input). Only
        # sanitize any residual inf across the full matrix; leave NaN in place
        # so LGBM treats raw NaN as missing exactly like in training.
        np.nan_to_num(X, copy=False, nan=np.nan, posinf=0.0, neginf=0.0)

        preds = np.zeros(X.shape[0], dtype=np.float64)
        if self.backend in ("xgb_mt", "cat_mt"):
            for clf, reg in zip(self.clf_boosters, self.reg_boosters):
                pred_prob = clf.predict_proba(X)[:, 1]
                pred_val = reg.predict(X)
                preds += pred_prob * pred_val
            preds /= len(self.reg_boosters)
        else:
            for booster in self.boosters:
                preds += booster.predict(X)
            preds /= len(self.boosters)

        if self.neutralize_alpha > 0.0 and self._neutralize_src_idx.size > 0:
            neutral_features = Xraw_full[:, self._neutralize_src_idx]
            preds = _neutralize_predictions(preds, neutral_features, alpha=self.neutralize_alpha)

        # Target rank transform: map predicted rank in [0,1] back to the
        # original target scale via the stored inverse-CDF LUT, BEFORE clipping
        # (the clip bound is in original scale).
        if self.target_transform == "rank" and self.target_quantile_lut is not None:
            preds = _inverse_cdf_map(preds, self.target_quantile_lut)

        clip = 3.0 * self.target_std
        preds = np.clip(preds, -clip, clip)
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0
        return preds

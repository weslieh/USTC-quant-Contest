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
    Vectorized over pairs (no Python per-pair loop).
    """
    X = np.where(np.isfinite(Xraw), Xraw.astype(np.float64), 0.0)
    a = X[:, pair_idx[:, 0]]  # (n, n_pairs)
    b = X[:, pair_idx[:, 1]]  # (n, n_pairs)
    mul = a * b
    div = a / (b + EPS)
    out = np.empty((X.shape[0], 2 * pair_idx.shape[0]), dtype=np.float32)
    out[:, 0::2] = mul
    out[:, 1::2] = div
    return out


def _neutralize_predictions(preds: np.ndarray, features: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Orthogonalize predictions against features (self-contained mirror of
    src.neutralization.neutralize_predictions).
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
    LUT (self-contained mirror of src.target_transform.inverse_cdf_map).
    """
    q = np.asarray(lut["q"], dtype=np.float64)
    v = np.asarray(lut["v"], dtype=np.float64)
    r = np.clip(np.asarray(rank_pred, dtype=np.float64), 0.0, 1.0)
    return np.interp(r, q, v).astype(np.float64)


def _cross_sectional(values, eps=EPS):
    """Per-time_id rank fraction / zscore / demean (mirrors training side)."""
    n = values.shape[0]
    out_rank = np.full(n, 0.5, dtype=np.float32)
    out_z = np.zeros(n, dtype=np.float32)
    out_dm = np.zeros(n, dtype=np.float32)
    if n < 2:
        return out_rank, out_z, out_dm
    vals = np.where(np.isfinite(values), values, 0.0).astype(np.float64)
    mean = vals.mean()
    std = vals.std(ddof=1) if n > 1 else 0.0
    rank_frac = (pd.Series(vals).rank(method="average").to_numpy() - 1.0) / (n - 1.0)
    z = np.where(std > eps, (vals - mean) / (std + eps), 0.0)
    dm = vals - mean
    return rank_frac.astype(np.float32), z.astype(np.float32), dm.astype(np.float32)


class _RollingBuf:
    """Per-asset rolling history (current row excluded), mirrors training."""

    def __init__(self, n_source, windows):
        self.n = n_source
        self.windows = list(windows)
        self.max_w = max(self.windows) if self.windows else 1
        self._b = {}

    def _get(self, aid):
        b = self._b.get(aid)
        if b is None:
            b = [np.zeros((self.max_w, self.n), dtype=np.float64), 0]
            self._b[aid] = b
        return b

    def compute(self, aid, row):
        arr, fill = self._get(aid)
        out = np.zeros(self.n * (1 + 2 * len(self.windows)), dtype=np.float32)
        if fill == 0:
            return out
        hist = arr[:fill]
        newest = hist[-1]
        o = 0
        for k in range(self.n):
            out[o] = newest[k]
            o += 1
            col = hist[:, k]
            for w in self.windows:
                win = col[-w:]
                out[o] = win.mean()
                o += 1
                out[o] = win.std(ddof=1) if win.size > 1 else 0.0
                o += 1
        return out

    def push(self, aid, row):
        arr, fill = self._get(aid)
        r = np.asarray(row, dtype=np.float64)
        if fill < self.max_w:
            arr[fill] = r
            self._b[aid][1] = fill + 1
        else:
            arr[:-1] = arr[1:]
            arr[-1] = r


class _SubModel:
    """One trained backend (lgb or xgb) loaded from its strategy dir."""

    def __init__(self, model_dir: Path, weight: float):
        self.dir = model_dir
        self.weight = weight
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        self.backend = meta.get("backend", "lgb")
        self.raw_feature_columns = list(meta["raw_feature_columns"])
        self.cs_source_columns = list(meta.get("cs_source_columns", []))
        self.rolling_source_columns = list(meta.get("rolling_source_columns", []))
        self.rolling_windows = list(meta.get("rolling_windows", []))

        self.boosters = []
        if self.backend == "xgb_mt":
            from xgboost import XGBRegressor, XGBClassifier
            self.clf_boosters = []
            self.reg_boosters = []
            for fname in meta.get("fold_files", []):
                clf = XGBClassifier()
                clf.load_model(str(model_dir / f"{fname}_clf.json"))
                self.clf_boosters.append(clf)
                reg = XGBRegressor()
                reg.load_model(str(model_dir / f"{fname}_reg.json"))
                self.reg_boosters.append(reg)
        elif self.backend == "cat_mt":
            from catboost import CatBoostRegressor, CatBoostClassifier
            self.clf_boosters = []
            self.reg_boosters = []
            for fname in meta.get("fold_files", []):
                clf = CatBoostClassifier()
                clf.load_model(str(model_dir / f"{fname}_clf.cbm"))
                self.clf_boosters.append(clf)
                reg = CatBoostRegressor()
                reg.load_model(str(model_dir / f"{fname}_reg.cbm"))
                self.reg_boosters.append(reg)
        elif self.backend == "xgb":
            from xgboost import XGBRegressor
            for fname in meta.get("fold_files", []):
                m = XGBRegressor()
                m.load_model(str(model_dir / fname))
                self.boosters.append(m)
        elif self.backend == "cat":
            from catboost import CatBoostRegressor
            for fname in meta.get("fold_files", []):
                m = CatBoostRegressor()
                m.load_model(str(model_dir / fname))
                self.boosters.append(m)
        else:
            for fname in meta.get("fold_files", []):
                self.boosters.append(lgb.Booster(model_file=str(model_dir / fname)))

    def predict(self, X):
        preds = np.zeros(X.shape[0], dtype=np.float64)
        if self.backend in ("xgb_mt", "cat_mt"):
            for clf, reg in zip(self.clf_boosters, self.reg_boosters):
                pred_prob = clf.predict_proba(X)[:, 1]
                pred_val = reg.predict(X)
                preds += pred_prob * pred_val
            return preds / len(self.reg_boosters)
        else:
            for b in self.boosters:
                preds += b.predict(X)
            return preds / len(self.boosters)


class Model:
    """Ensemble of two trained backends (e.g. LightGBM + XGBoost fold ensembles).

    Both sub-models MUST share the same engineered-feature spec (same raw +
    cross-sectional + rolling columns) so features are built once and fed to
    both. Config comes from this dir's ensemble_meta.json:
        {"models": [{"dir": "strategy_lgb", "weight": 0.6},
                    {"dir": "strategy_xgb", "weight": 0.4}]}
    Relative ``dir`` paths resolve against this ensemble dir's parent.
    """

    def __init__(self):
        here = Path(__file__).resolve().parent
        em = json.loads((here / "ensemble_meta.json").read_text(encoding="utf-8"))
        self.subs = []
        for spec in em["models"]:
            d = Path(spec["dir"])
            if not d.is_absolute():
                d = (here.parent / d).resolve()
            self.subs.append(_SubModel(d, float(spec.get("weight", 1.0))))
        # Feature spec from the first sub-model (all subs must agree).
        s0 = self.subs[0]
        self.raw_feature_columns = s0.raw_feature_columns
        self.cs_source_columns = s0.cs_source_columns
        self.rolling_source_columns = s0.rolling_source_columns
        self.rolling_windows = s0.rolling_windows
        self._cs_src_idx = np.asarray(
            [self.raw_feature_columns.index(c) for c in self.cs_source_columns
             if c in self.raw_feature_columns], dtype=np.intp
        )
        self._roll_src_idx = np.asarray(
            [self.raw_feature_columns.index(c) for c in self.rolling_source_columns
             if c in self.raw_feature_columns], dtype=np.intp
        )
        self.target_std = float(
            json.loads((self.subs[0].dir / "model_meta.json").read_text(encoding="utf-8"))
            .get("target_std") or 1.0
        )
        # Target rank transform (shared spec from subs[0]).
        self.target_transform = s0_meta.get("target_transform", "none")
        self.target_quantile_lut = s0_meta.get("target_quantile_lut") or None
        s0_meta = json.loads((s0.dir / "model_meta.json").read_text(encoding="utf-8"))
        self.neutralize_alpha = float(s0_meta.get("neutralize_alpha", 0.0))
        self.neutralize_features = s0_meta.get("neutralize_features", [])
        self._neutralize_src_idx = np.asarray(
            [self.raw_feature_columns.index(c) for c in self.neutralize_features
             if c in self.raw_feature_columns], dtype=np.intp
        )
        # Interaction spec (shared by all subs).
        self.interaction_pairs = [tuple(p) for p in s0_meta.get("interaction_source_columns", [])]
        self._inter_pair_idx = np.asarray(
            [[self.raw_feature_columns.index(a), self.raw_feature_columns.index(b)]
             for a, b in self.interaction_pairs
             if a in self.raw_feature_columns and b in self.raw_feature_columns],
            dtype=np.intp,
        ).reshape(-1, 2)
        self.rolling = _RollingBuf(len(self._roll_src_idx), self.rolling_windows)
        self.last_time_id: int | None = None

    def _build_features(self, test: pd.DataFrame) -> np.ndarray:
        asset_ids = test["asset_id"].to_numpy()
        Xraw = test.to_numpy(dtype=np.float32, copy=True)
        col_pos = test.columns.get_indexer(self.raw_feature_columns)
        Xraw = Xraw[:, col_pos]
        np.nan_to_num(Xraw, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
        n = Xraw.shape[0]
        feats = [Xraw]
        if self._cs_src_idx.size:
            cs_block = np.zeros((n, self._cs_src_idx.size * 3), dtype=np.float32)
            for k, ci in enumerate(self._cs_src_idx):
                r, z, dm = _cross_sectional(Xraw[:, ci])
                cs_block[:, 3 * k] = r
                cs_block[:, 3 * k + 1] = z
                cs_block[:, 3 * k + 2] = dm
            feats.append(cs_block)
        if self._roll_src_idx.size and self.rolling_windows:
            block_w = 1 + 2 * len(self.rolling_windows)
            roll_block = np.zeros((n, self._roll_src_idx.size * block_w), dtype=np.float32)
            for i in range(n):
                aid = int(asset_ids[i])
                roll_block[i] = self.rolling.compute(aid, Xraw[i, self._roll_src_idx])
            for i in range(n):
                self.rolling.push(int(asset_ids[i]), Xraw[i, self._roll_src_idx])
            feats.append(roll_block)
        # Interactions: within-row pairwise mul/div (feature axis), no state.
        if self._inter_pair_idx.size:
            feats.append(_interaction_block(Xraw, self._inter_pair_idx))
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
        np.nan_to_num(X, copy=False, nan=np.nan, posinf=0.0, neginf=0.0)
        total_w = sum(s.weight for s in self.subs) or 1.0
        preds = np.zeros(X.shape[0], dtype=np.float64)
        for s in self.subs:
            preds += s.weight * s.predict(X)
        preds /= total_w

        if self.neutralize_alpha > 0.0 and self._neutralize_src_idx.size > 0:
            neutral_features = Xraw_full[:, self._neutralize_src_idx]
            preds = _neutralize_predictions(preds, neutral_features, alpha=self.neutralize_alpha)

        # Target rank transform: map predicted rank -> target scale before clip.
        if self.target_transform == "rank" and self.target_quantile_lut is not None:
            preds = _inverse_cdf_map(preds, self.target_quantile_lut)

        clip = 3.0 * self.target_std
        preds = np.clip(preds, -clip, clip)
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0
        return preds

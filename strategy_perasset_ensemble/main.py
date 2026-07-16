"""Per-asset three-backend ensemble for the Time-Series API.

Loads several per-asset model directories (each trained by ``train.py
--per-asset --backend {lgb,xgb,cat}``), routes each row by asset_id to that
asset's fold ensemble within each sub-model, then averages the sub-models
weight-normalized. Output is clipped to ``±3*target_std`` (shared).

Config (ensemble_meta.json in this dir):
    {"models": [{"dir": "strategy_perasset_lgb", "weight": 1.0},
                {"dir": "strategy_perasset_xgb", "weight": 1.0},
                {"dir": "strategy_perasset_cat", "weight": 1.0}]}

Each sub-dir must be a per-asset model (model_meta.json with per_asset=true
and asset_dirs). Relative ``dir`` paths resolve against this dir's parent
(the project root).

Self-contained: no ``src.*`` imports (submission package only has the
strategy dir on sys.path).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_booster(base_backend: str, fold_ext: str, fpath: Path):
    if base_backend == "xgb":
        from xgboost import XGBRegressor
        m = XGBRegressor()
        m.load_model(str(fpath))
        return m
    elif base_backend == "cat":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(str(fpath))
        return m
    else:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(fpath))


class _PerAssetSub:
    """One per-asset backend: asset_id -> list of fold boosters."""

    def __init__(self, model_dir: Path, weight: float):
        self.weight = weight
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        # base_backend / fold_ext: prefer explicit fields, else infer from the
        # ``backend`` tag (e.g. "lgb_perasset"). Older LGB per-asset metas had
        # base_backend=None / fold_ext=None, so guard against that.
        be = meta.get("base_backend")
        if not be:
            tag = meta.get("backend", "lgb_perasset")
            be = tag.split("_perasset")[0] if "_perasset" in tag else "lgb"
        self.base_backend = be
        ext_map = {"lgb": "txt", "xgb": "json", "cat": "cbm"}
        self.fold_ext = meta.get("fold_ext") or ext_map.get(self.base_backend, "txt")
        self.asset_models: dict[int, list] = {}
        for spec in meta.get("asset_dirs", []):
            aid = int(spec["asset_id"])
            adir = model_dir / spec["dir"]
            boosters = [
                _load_booster(self.base_backend, self.fold_ext,
                              adir / f"booster_fold_{f}.{self.fold_ext}")
                for f in range(int(spec["n_folds"]))
            ]
            self.asset_models[aid] = boosters

    def predict_asset(self, aid: int, Xa: np.ndarray) -> np.ndarray:
        boosters = self.asset_models.get(aid)
        if not boosters:
            return np.zeros(Xa.shape[0], dtype=np.float64)
        pa = np.zeros(Xa.shape[0], dtype=np.float64)
        for b in boosters:
            pa += b.predict(Xa)
        return pa / len(boosters)


class Model:
    """Weighted ensemble of per-asset backends, routed by asset_id."""

    def __init__(self):
        here = Path(__file__).resolve().parent
        em = json.loads((here / "ensemble_meta.json").read_text(encoding="utf-8"))
        self.subs = []
        for spec in em["models"]:
            d = Path(spec["dir"])
            if not d.is_absolute():
                d = (here.parent / d).resolve()
            self.subs.append(_PerAssetSub(d, float(spec.get("weight", 1.0))))
        # Shared feature spec (all subs must agree) + clip bound, from sub[0].
        first_dir = Path(em["models"][0]["dir"])
        if not first_dir.is_absolute():
            first_dir = (here.parent / first_dir).resolve()
        s0 = json.loads((first_dir / "model_meta.json").read_text(encoding="utf-8"))
        self.raw_feature_columns = list(s0["raw_feature_columns"])
        self.target_std = float(s0.get("target_std") or 1.0)
        self.last_time_id: int | None = None

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must strictly increase in Time-Series API order")
        self.last_time_id = time_id

        asset_ids = test["asset_id"].to_numpy()
        col_pos = test.columns.get_indexer(self.raw_feature_columns)
        Xraw = test.to_numpy(dtype=np.float32, copy=True)[:, col_pos]
        np.nan_to_num(Xraw, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)

        total_w = sum(s.weight for s in self.subs) or 1.0
        preds = np.zeros(Xraw.shape[0], dtype=np.float64)
        # Route by asset_id; for each asset, each sub predicts, weight-averaged.
        for aid in np.unique(asset_ids):
            mask = asset_ids == aid
            if not mask.any():
                continue
            Xa = Xraw[mask]
            aid_int = int(aid)
            sub_avg = np.zeros(Xa.shape[0], dtype=np.float64)
            for s in self.subs:
                sub_avg += s.weight * s.predict_asset(aid_int, Xa)
            preds[mask] = sub_avg
        preds /= total_w

        clip = 3.0 * self.target_std
        preds = np.clip(preds, -clip, clip)
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0
        return preds

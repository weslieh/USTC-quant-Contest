"""Per-asset inference model for the Time-Series API.

Loads one set of fold boosters per asset_id (trained by ``train.py
--per-asset``) and routes each row to its asset's model. ``predict(test)``
receives one time_id slice (~15 rows, one per asset), groups by asset_id,
predicts each group with that asset's 5-fold ensemble, and reassembles the
predictions in the original row order. Output is clipped to
``±3*target_std`` (shared across assets, original target scale).

Self-contained: no ``src.*`` imports (the submission package only has the
strategy dir on sys.path).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class Model:
    """Ensemble of per-asset fold boosters, routed by asset_id."""

    def __init__(self):
        here = Path(__file__).resolve().parent
        meta = json.loads((here / "model_meta.json").read_text(encoding="utf-8"))

        self.raw_feature_columns = list(meta["raw_feature_columns"])
        self.target_std = float(meta.get("target_std") or 1.0)
        self.base_backend = meta.get("base_backend", "lgb")
        self.fold_ext = meta.get("fold_ext", "txt")

        # asset_id -> list of fold boosters
        self.asset_models: dict[int, list] = {}
        for spec in meta.get("asset_dirs", []):
            aid = int(spec["asset_id"])
            adir = here / spec["dir"]
            boosters = []
            for fold_idx in range(int(spec["n_folds"])):
                fpath = adir / f"booster_fold_{fold_idx}.{self.fold_ext}"
                boosters.append(self._load_booster(fpath))
            self.asset_models[aid] = boosters

        # Fallback: if an asset has no model (shouldn't happen), predict 0.
        self.last_time_id: int | None = None

    def _load_booster(self, fpath: Path):
        """Load one fold booster per backend (self-contained, no src import)."""
        if self.base_backend == "xgb":
            from xgboost import XGBRegressor
            m = XGBRegressor()
            m.load_model(str(fpath))
            return m
        elif self.base_backend == "cat":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(str(fpath))
            return m
        else:
            import lightgbm as lgb
            return lgb.Booster(model_file=str(fpath))

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must strictly increase in Time-Series API order")
        self.last_time_id = time_id

        asset_ids = test["asset_id"].to_numpy()
        col_pos = test.columns.get_indexer(self.raw_feature_columns)
        Xraw = test.to_numpy(dtype=np.float32, copy=True)[:, col_pos]
        # Keep raw NaN (LGBM native missing-value handling), sanitize inf.
        np.nan_to_num(Xraw, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)

        preds = np.zeros(Xraw.shape[0], dtype=np.float64)
        for aid, boosters in self.asset_models.items():
            mask = asset_ids == aid
            if not mask.any():
                continue
            Xa = Xraw[mask]
            pa = np.zeros(Xa.shape[0], dtype=np.float64)
            for b in boosters:
                pa += b.predict(Xa)
            pa /= len(boosters)
            preds[mask] = pa

        clip = 3.0 * self.target_std
        preds = np.clip(preds, -clip, clip)
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0
        return preds

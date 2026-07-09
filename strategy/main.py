from __future__ import annotations

import json
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
    """

    def __init__(self):
        here = Path(__file__).resolve().parent
        model_path = here / "model.txt"
        meta_path = here / "model_meta.json"

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_columns = list(meta["feature_columns"])

        self.booster = lgb.Booster(model_file=str(model_path))
        self.last_time_id: int | None = None

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must strictly increase in Time-Series API order")
        self.last_time_id = time_id

        # Build the feature matrix in the exact column order the booster was
        # trained on. to_numpy may return a read-only / non-writable view of
        # the pandas buffer, so materialize a writable copy before sanitizing.
        X = np.array(test.loc[:, self.feature_columns].to_numpy(), dtype=np.float32, copy=True)
        # LightGBM handles NaN natively; sanitize only inf which it does not.
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        preds = self.booster.predict(X)
        preds = np.asarray(preds, dtype=np.float64).reshape(-1)
        # Guarantee finite output (safety net for the eval API).
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0
        return preds

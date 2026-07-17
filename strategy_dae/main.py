"""DAE+MLP inference model for the Time-Series API.

Self-contained: defines the model architecture inline (must match
train_dae.py's DAEMLP) and loads torch state_dicts. predict(test) builds raw
features, runs the encoder+head per fold, averages fold predictions, clips to
±3*target_std. asset_id embedding is a fixed per-asset learned vector.

No src.* imports (submission package only has the strategy dir on sys.path).
Requires torch at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _swish(x):
    return torch.nn.functional.silu(x)


class _DAEMLP(nn.Module):
    def __init__(self, n_features=323, n_assets=15, asset_emb_dim=4,
                 hidden=128, bottleneck=32, head_hidden=64):
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, asset_emb_dim)
        in_dim = n_features + asset_emb_dim
        self.ln_in = nn.LayerNorm(in_dim)
        self.enc_fc1 = nn.Linear(in_dim, hidden)
        self.ln_e1 = nn.LayerNorm(hidden)
        self.enc_fc2 = nn.Linear(hidden, bottleneck)
        self.ln_b = nn.LayerNorm(bottleneck)
        self.dec_fc1 = nn.Linear(bottleneck, hidden)
        self.dec_fc2 = nn.Linear(hidden, n_features)
        self.head_fc1 = nn.Linear(bottleneck, head_hidden)
        self.ln_h = nn.LayerNorm(head_hidden)
        self.head_fc2 = nn.Linear(head_hidden, 1)

    def forward(self, x_raw, asset_id):
        a = self.asset_emb(asset_id)
        x = torch.cat([x_raw, a], dim=1)
        x = self.ln_in(x)
        x = _swish(self.ln_e1(self.enc_fc1(x)))
        z = self.ln_b(self.enc_fc2(x))
        h = _swish(self.ln_h(self.head_fc1(z)))
        pred = self.head_fc2(h).squeeze(-1)
        return pred


class Model:
    """Ensemble of DAE+MLP fold models."""

    def __init__(self):
        here = Path(__file__).resolve().parent
        meta = json.loads((here / "model_meta.json").read_text(encoding="utf-8"))
        self.raw_feature_columns = list(meta["raw_feature_columns"])
        self.target_std = float(meta.get("target_std") or 1.0)
        self.device = torch.device("cpu")
        n_features = meta["n_features"]
        n_assets = meta["n_assets"]
        asset_emb_dim = meta["asset_emb_dim"]
        hidden = meta["hidden"]
        bottleneck = meta["bottleneck"]
        self.models = []
        for fname in meta.get("fold_files", []):
            m = _DAEMLP(n_features=n_features, n_assets=n_assets,
                        asset_emb_dim=asset_emb_dim, hidden=hidden,
                        bottleneck=bottleneck)
            state = torch.load(here / fname, map_location=self.device, weights_only=True)
            m.load_state_dict(state)
            m.eval()
            self.models.append(m)
        self.last_time_id: int | None = None

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        time_id = int(test["time_id"].iloc[0])
        if self.last_time_id is not None and time_id <= self.last_time_id:
            raise ValueError("time_id must strictly increase in Time-Series API order")
        self.last_time_id = time_id

        col_pos = test.columns.get_indexer(self.raw_feature_columns)
        Xraw = test.to_numpy(dtype=np.float32, copy=True)[:, col_pos]
        Xraw = np.nan_to_num(Xraw, nan=0.0, posinf=0.0, neginf=0.0)
        asset_ids = test["asset_id"].to_numpy().astype(np.int64)

        Xt = torch.from_numpy(Xraw)
        aidt = torch.from_numpy(asset_ids)
        preds = np.zeros(Xraw.shape[0], dtype=np.float64)
        with torch.no_grad():
            for m in self.models:
                preds += m(Xt, aidt).numpy()
        preds /= len(self.models)

        clip = 3.0 * self.target_std
        preds = np.clip(preds, -clip, clip)
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0
        return preds

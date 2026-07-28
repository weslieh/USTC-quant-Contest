"""Large supervised NN inference model for the Time-Series API.

Self-contained: defines the model architecture inline (must match train_nn.py's
TabularNN) and loads torch state_dicts. predict(test) builds raw features, runs
the network per fold, averages fold predictions, clips to +/-3*target_std.

No src.* imports (submission package only has the strategy dir on sys.path).
Requires torch at inference time. Row-wise (PeriodicEmbedding + LayerNorm, no
cross-sample dependency) so eval-mode inference is drift-safe.

Usage (public-LB CSV, no inference limits):
  cp strategy_nn/main.py <model_dir>/main.py   # or run directly from strategy_nn/
  python timeseries_api/run_timeseries_api.py --strategy-dir strategy_nn --output out/sub_nn.csv --split test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _PeriodicFeatureEmbedding(nn.Module):
    def __init__(self, n_features, n_periodic=16, emb_dim=8):
        super().__init__()
        self.n_features = n_features
        self.n_periodic = n_periodic
        self.emb_dim = emb_dim
        self.freq = nn.Parameter(torch.randn(n_features, n_periodic) * 0.1)
        self.phase = nn.Parameter(torch.zeros(n_features, n_periodic))
        self.mix = nn.Parameter(torch.randn(2 * n_periodic, emb_dim) * 0.02)

    def forward(self, x):
        proj = x.unsqueeze(-1) * self.freq.unsqueeze(0) + self.phase.unsqueeze(0)
        emb = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        emb = emb @ self.mix
        return emb.reshape(emb.shape[0], -1)


class _ResMLP(nn.Module):
    def __init__(self, dim, n_blocks=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.Dropout(dropout),
            ))
        self.act = nn.SiLU()

    def forward(self, x):
        for blk in self.blocks:
            x = x + self.act(blk(x))
        return x


class _TabularNN(nn.Module):
    def __init__(self, n_features=323, n_assets=15, n_periodic=16, feat_emb_dim=8,
                 asset_emb_dim=8, hidden=256, n_blocks=4, dropout=0.1):
        super().__init__()
        self.feat_emb = _PeriodicFeatureEmbedding(n_features, n_periodic, feat_emb_dim)
        in_dim = n_features * feat_emb_dim + asset_emb_dim
        self.asset_emb = nn.Embedding(n_assets, asset_emb_dim)
        self.ln_in = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        self.trunk = _ResMLP(hidden, n_blocks, dropout)
        self.ln_out = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_raw, asset_id):
        fe = self.feat_emb(x_raw)
        ae = self.asset_emb(asset_id)
        h = torch.cat([fe, ae], dim=-1)
        h = self.ln_in(h)
        h = self.proj(h)
        h = self.trunk(h)
        h = self.ln_out(h)
        pred = self.head(h).squeeze(-1)
        return pred


class Model:
    """Ensemble of TabularNN fold models."""

    def __init__(self):
        here = Path(__file__).resolve().parent
        meta = json.loads((here / "model_meta.json").read_text(encoding="utf-8"))
        self.raw_feature_columns = list(meta["raw_feature_columns"])
        self.target_std = float(meta.get("target_std") or 1.0)
        self.device = torch.device("cpu")
        self.models = []
        for fname in meta.get("fold_files", []):
            m = _TabularNN(
                n_features=meta["n_features"], n_assets=meta["n_assets"],
                n_periodic=meta["n_periodic"], feat_emb_dim=meta["feat_emb_dim"],
                asset_emb_dim=meta["asset_emb_dim"], hidden=meta["hidden"],
                n_blocks=meta["n_blocks"], dropout=meta.get("dropout", 0.1),
            )
            try:
                state = torch.load(here / fname, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(here / fname, map_location=self.device)
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
        preds /= max(1, len(self.models))

        clip = 3.0 * self.target_std
        preds = np.clip(preds, -clip, clip)
        bad = ~np.isfinite(preds)
        if bad.any():
            preds[bad] = 0.0
        return preds

"""Large supervised NN for target prediction (big-NN signal probe).

Motivation: the per-asset GBDT (0.0031) is at the GBDT axis-aligned-split
ceiling. A larger NN with periodic feature embeddings (PLR, shown to let tabular
NNs match/beat GBDT, ICLR 2025) may extract weak signal more efficiently than
axis-aligned trees. This is a SIGNAL PROBE: train on GPU (public-LB CSV phase
has no inference limits), check whether CV can approach/exceed the GBDT's ~0.002
per-fold. DAE (small 2-layer autoencoder, reconstruction loss) scored 0.000065,
so this deliberately differs: (1) no reconstruction — full capacity on target,
(2) periodic feature embeddings instead of raw->Linear, (3) deeper residual MLP,
(4) LayerNorm (row-wise, avoids the BN cross-sample death) + asset embedding
(the one signal class that survived).

If CV > ~0.001 (vs DAE 0.000065, GBDT ~0.002), the NN has signal → public-LB CSV
submission to confirm, then distill to GBDT or lightweight NN for private-LB.
If CV ~ DAE (0.0001), NN route is truly dead.

Architecture:
  x_raw (n_feat=323) -> PeriodicEmbedding (learnable freqs, n_periodic=16)
      -> (n_feat * 2*n_periodic) concat with asset_emb(15->8)
      -> LayerNorm -> [Linear(->256) -> LN -> Swish -> Dropout -> residual] x 4
      -> Linear(256->1)
  loss = sum(w*(pred-y)^2)/sum(w)  (weighted MSE, matches R^2 direction)

Usage (GPU):
  python train_nn.py --data-root data --partitions 1 --n-folds 3 --embargo 5000 \
      --epochs 40 --batch-size 4096 --lr 1e-3 --out-dir strategy_nn_pilot --fresh
  # full:
  python train_nn.py --data-root data --partitions 9 --n-folds 5 --embargo 5000 \
      --epochs 40 --batch-size 8192 --out-dir strategy_nn --save-model --fresh
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.dataset import load_train
from src.cv import time_cv_split
from src.features import get_feature_columns
from src.interactions import build_per_asset_features, per_asset_column_names
from src.metrics import weighted_zero_mean_r2


def swish(x):
    return x * torch.sigmoid(x)


class PeriodicFeatureEmbedding(nn.Module):
    """Periodic (PLR-style) embedding for continuous features.

    For each feature, project x through learnable frequencies via cos/sin:
      e(x) = concat[cos(x * f_1), sin(x * f_1), ..., cos(x * f_P), sin(x * f_P)]
    then mix with a per-feature linear layer. This lets the NN learn a smooth
    nonlinear basis per feature — the key ingredient that lets tabular NNs
    compete with GBDT (Gorishniy et al., ICLR 2025). Shared across rows, so
    it is row-wise (no cross-sample dependency) -> drift-safe at inference.
    """

    def __init__(self, n_features, n_periodic=16, emb_dim=8):
        super().__init__()
        self.n_features = n_features
        self.n_periodic = n_periodic
        self.emb_dim = emb_dim
        # learnable frequencies (initialized ~ N(0,0.1)); shape (n_features, n_periodic)
        self.freq = nn.Parameter(torch.randn(n_features, n_periodic) * 0.1)
        self.phase = nn.Parameter(torch.zeros(n_features, n_periodic))
        # per-feature linear: 2*n_periodic -> emb_dim (applied independently per feature)
        self.mix = nn.Parameter(torch.randn(2 * n_periodic, emb_dim) * 0.02)

    def forward(self, x):
        # x: (B, n_features)
        proj = x.unsqueeze(-1) * self.freq.unsqueeze(0) + self.phase.unsqueeze(0)  # (B, n_feat, P)
        emb = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (B, n_feat, 2P)
        # per-feature mix: (B, n_feat, 2P) @ (2P, emb_dim) -> (B, n_feat, emb_dim)
        emb = emb @ self.mix
        return emb.reshape(emb.shape[0], -1)  # (B, n_feat*emb_dim)


class ResMLP(nn.Module):
    def __init__(self, dim, n_blocks=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.Dropout(dropout),
            ))
        self.act = nn.SiLU()  # Swish

    def forward(self, x):
        for blk in self.blocks:
            x = x + self.act(blk(x))  # pre-act residual
        return x


class TabularNN(nn.Module):
    def __init__(self, n_features=323, n_assets=15, n_periodic=16, feat_emb_dim=8,
                 asset_emb_dim=8, hidden=256, n_blocks=4, dropout=0.1):
        super().__init__()
        self.feat_emb = PeriodicFeatureEmbedding(n_features, n_periodic, feat_emb_dim)
        in_dim = n_features * feat_emb_dim + asset_emb_dim
        self.asset_emb = nn.Embedding(n_assets, asset_emb_dim)
        self.ln_in = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden)
        self.trunk = ResMLP(hidden, n_blocks, dropout)
        self.ln_out = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_raw, asset_id):
        fe = self.feat_emb(x_raw)  # (B, n_feat*emb_dim)
        ae = self.asset_emb(asset_id)  # (B, asset_emb_dim)
        h = torch.cat([fe, ae], dim=-1)
        h = self.ln_in(h)
        h = self.proj(h)
        h = self.trunk(h)
        h = self.ln_out(h)
        return self.head(h).squeeze(-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--partitions", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--embargo", type=int, default=5000)
    p.add_argument("--sample-rows", type=int, default=0, help="Cap train rows per fold (0=all).")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--per-asset-topk", type=int, default=0, help="Per-asset masked feature columns (0=off). Same scheme as train.py: each asset's top-K raw features (by --per-asset-importance-csv) become a column = feature value on that asset's rows, 0 elsewhere. Fed as extra input features to the NN.")
    p.add_argument("--per-asset-importance-csv", type=str, default="out/eda_full/m3_per_asset_importance.csv")
    p.add_argument("--n-periodic", type=int, default=16)
    p.add_argument("--feat-emb-dim", type=int, default=8)
    p.add_argument("--asset-emb-dim", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0, help="Adam L2 weight decay (the single-partition probe overfits fast — epoch 0-1 peak then declines; try 1e-4..1e-3 to regularize).")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="strategy_nn")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--fresh", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_dir = out_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cv_dir / "scores.json"

    print("Loading data ...")
    frame = load_train(args.data_root, partitions=args.partitions)
    raw_feature_cols = get_feature_columns(frame)
    n_features = len(raw_feature_cols)
    n_assets = 15
    print(f"  raw features: {n_features}")

    # Per-asset masked feature specs (optional, same scheme as train.py).
    per_asset_specs = []
    pa_col_names = []
    if args.per_asset_topk > 0:
        from train import load_per_asset_specs
        per_asset_specs = load_per_asset_specs(
            args.per_asset_importance_csv, raw_feature_cols, args.per_asset_topk)
        pa_col_names = per_asset_column_names(per_asset_specs)
        n_features = n_features + len(pa_col_names)
        print(f"  per-asset topk={args.per_asset_topk}: +{len(pa_col_names)} cols -> {n_features} total input features")

    folds = time_cv_split(frame, n_folds=args.n_folds, valid_frac=args.valid_frac, embargo=args.embargo)
    print(f"  folds: {len(folds)}")

    # resume support
    cached = {}
    if (not args.fresh) and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"  resume: {len(cached.get('scores', []))} fold(s) done")
        except Exception:
            cached = {}
    if args.fresh:
        cached = {}
    target_std = cached.get("target_std")
    done_scores = list(cached.get("scores", []))
    scores = []
    fold_files = []

    for fold_idx, (train_lf, valid_lf) in enumerate(folds):
        fname = f"booster_fold_{fold_idx}.pt"
        if fold_idx < len(done_scores) and (out_dir / fname).exists():
            print(f"\n--- Fold {fold_idx + 1} / {len(folds)} (cached, score={done_scores[fold_idx]:.6f}) ---")
            fold_files.append(fname)
            scores.append(float(done_scores[fold_idx]))
            continue

        print(f"\n--- Fold {fold_idx + 1} / {len(folds)} ---")
        train_df = train_lf.collect()
        valid_df = valid_lf.collect()
        if args.sample_rows > 0 and len(train_df) > args.sample_rows:
            train_df = train_df.head(args.sample_rows)
        print(f"  train rows: {len(train_df)}  valid rows: {len(valid_df)}")

        X_tr = np.nan_to_num(train_df.select(raw_feature_cols).to_numpy().astype(np.float32))
        X_va = np.nan_to_num(valid_df.select(raw_feature_cols).to_numpy().astype(np.float32))
        if per_asset_specs:
            train_df_pa, _ = build_per_asset_features(train_df, per_asset_specs)
            valid_df_pa, _ = build_per_asset_features(valid_df, per_asset_specs)
            pa_tr = np.nan_to_num(train_df_pa.select(pa_col_names).to_numpy().astype(np.float32))
            pa_va = np.nan_to_num(valid_df_pa.select(pa_col_names).to_numpy().astype(np.float32))
            X_tr = np.hstack([X_tr, pa_tr])
            X_va = np.hstack([X_va, pa_va])
        aid_tr = train_df["asset_id"].to_numpy().astype(np.int64)
        aid_va = valid_df["asset_id"].to_numpy().astype(np.int64)
        y_tr = train_df["target"].to_numpy().astype(np.float32)
        y_va = valid_df["target"].to_numpy().astype(np.float32)
        w_tr = train_df["weight"].to_numpy().astype(np.float32)
        w_va = valid_df["weight"].to_numpy().astype(np.float32)
        if target_std is None:
            target_std = float(np.std(y_tr))

        Xtr_t = torch.tensor(X_tr, device=device)
        aidtr_t = torch.tensor(aid_tr, device=device)
        ytr_t = torch.tensor(y_tr, device=device)
        wtr_t = torch.tensor(w_tr, device=device)
        Xva_t = torch.tensor(X_va, device=device)
        aidva_t = torch.tensor(aid_va, device=device)
        yva_t = torch.tensor(y_va, device=device)
        wva_t = torch.tensor(w_va, device=device)

        model = TabularNN(n_features=n_features, n_assets=n_assets,
                          n_periodic=args.n_periodic, feat_emb_dim=args.feat_emb_dim,
                          asset_emb_dim=args.asset_emb_dim, hidden=args.hidden,
                          n_blocks=args.n_blocks, dropout=args.dropout).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if fold_idx == 0:
            print(f"  model params: {n_params:,}")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        n = Xtr_t.shape[0]
        n_batches = math.ceil(n / args.batch_size)
        best_score = -1e9
        best_state = None
        best_epoch = -1
        bad_epochs = 0

        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            tot_loss = 0.0
            for b in range(n_batches):
                idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
                xb = Xtr_t[idx]; aidb = aidtr_t[idx]; yb = ytr_t[idx]; wb = wtr_t[idx]
                opt.zero_grad()
                pred = model(xb, aidb)
                l = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp(min=1e-8)
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot_loss += float(l.detach())
            sched.step()
            model.eval()
            with torch.no_grad():
                # batch eval to bound memory on large valid
                preds = []
                for i in range(0, Xva_t.shape[0], 65536):
                    preds.append(model(Xva_t[i:i + 65536], aidva_t[i:i + 65536]).cpu().numpy())
                pred_va_np = np.concatenate(preds)
            score = weighted_zero_mean_r2(y_va, pred_va_np, w_va)
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            if epoch % 2 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch}: train_loss={tot_loss/n_batches:.4f} "
                      f"valid_r2={score:.6f} (best {best_score:.6f}@{best_epoch})")
            if bad_epochs >= args.patience:
                print(f"  early stop at epoch {epoch}, best {best_score:.6f}@{best_epoch}")
                break

        if args.save_model:
            torch.save(best_state, out_dir / fname)
        fold_files.append(fname)
        scores.append(float(best_score))
        print(f"  fold {fold_idx} best valid R2: {best_score:.6f}")
        cache_path.write_text(json.dumps({"scores": scores, "target_std": target_std}, indent=2), encoding="utf-8")
        del Xtr_t, aidtr_t, ytr_t, wtr_t, Xva_t, aidva_t, yva_t, wva_t, model, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nCV scores: {[f'{s:.6f}' for s in scores]}")
    print(f"Mean CV: {float(np.mean(scores)):.6f}  Std: {float(np.std(scores)):.6f}")
    print(f"\n[signal probe] DAE was 0.000065, GBDT per-fold ~0.002. "
          f"This NN: {float(np.mean(scores)):.6f}")

    if args.save_model:
        meta = {
            "backend": "nn",
            "raw_feature_columns": list(raw_feature_cols),
            "n_features": n_features,
            "n_assets": n_assets,
            "n_periodic": args.n_periodic,
            "feat_emb_dim": args.feat_emb_dim,
            "asset_emb_dim": args.asset_emb_dim,
            "hidden": args.hidden,
            "n_blocks": args.n_blocks,
            "per_asset_specs": [[a, f] for a, f in per_asset_specs],
            "per_asset_feature_columns": pa_col_names,
            "n_folds": len(folds),
            "fold_files": fold_files,
            "target_std": target_std,
            "cv_mean": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
        }
        (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved meta -> {out_dir / 'model_meta.json'}")


if __name__ == "__main__":
    main()

"""Train a supervised denoising autoencoder + MLP for target prediction.

Architecture (per-row, no cross-sample/cross-time dependence):
  x = [323 raw features] + [asset_id embedding(15->4)]
  encoder: LayerNorm -> GaussianNoise -> Linear(327->128) -> LayerNorm -> Swish -> Linear(128->32)  (bottleneck z)
  decoder: Linear(32->128) -> Swish -> Linear(128->323)               (reconstruct x_raw)
  head:    Linear(32->64) -> LayerNorm -> Swish -> Linear(64->1)        (predict target)

Joint loss = recon_MSE(x_raw) + lambda * weighted_target_MSE(z->head).

LayerNorm (not BatchNorm): per-row normalization, no batch dependence at
inference (BN train-mode == cross-sectional dependency = the dead cs path;
BN eval-mode running stats break under AUC=1.0 drift). asset_id embedding is
a fixed per-asset learned vector (the one signal class that survived: a fixed
per-row attribute, not a train-distribution derivative).

Trained on GPU if available, saved as torch state_dict per fold; inference is
torch eval-mode forward (self-contained strategy_dae/main.py).

Usage:
  python train_dae.py --data-root data --partitions 9 --n-folds 5 --embargo 5000 \
      --epochs 30 --batch-size 4096 --lr 1e-3 --out-dir strategy_dae --fresh
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import load_train
from src.cv import time_cv_split
from src.metrics import weighted_zero_mean_r2
from src.features import get_feature_columns


class DAEMLP(nn.Module):
    """Supervised denoising autoencoder + MLP target head (per-row)."""

    def __init__(self, n_features=323, n_assets=15, asset_emb_dim=4,
                 hidden=128, bottleneck=32, head_hidden=64, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
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

    def encode(self, x_raw, asset_id):
        a = self.asset_emb(asset_id)  # (N, emb)
        x = torch.cat([x_raw, a], dim=1)
        x = self.ln_in(x)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = swish(self.ln_e1(self.enc_fc1(x)))
        z = self.ln_b(self.enc_fc2(x))
        return z

    def decode(self, z):
        h = swish(self.dec_fc1(z))
        recon = self.dec_fc2(h)
        return recon

    def forward(self, x_raw, asset_id):
        z = self.encode(x_raw, asset_id)
        recon = self.decode(z)
        h = swish(self.ln_h(self.head_fc1(z)))
        pred = self.head_fc2(h).squeeze(-1)
        return recon, pred, z


def swish(x):
    return torch.nn.functional.silu(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--partitions", type=int, default=9)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--embargo", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--recon-weight", type=float, default=1.0, help="weight on reconstruction loss")
    p.add_argument("--target-weight", type=float, default=1.0, help="weight on target loss")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--asset-emb-dim", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="strategy_dae")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--sample-rows", type=int, default=0, help="if >0, cap train rows per fold (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "model_meta.json"
    cv_dir = out_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cv_dir / "scores.json"

    print("Loading data ...")
    frame = load_train(args.data_root, partitions=args.partitions)
    raw_feature_cols = get_feature_columns(frame)
    print(f"  raw features: {len(raw_feature_cols)}")

    folds = time_cv_split(frame, n_folds=args.n_folds, valid_frac=args.valid_frac, embargo=args.embargo)
    print(f"  folds: {len(folds)}")

    n_features = len(raw_feature_cols)
    n_assets = 15
    fold_files = []
    scores = []
    target_std = None

    # Resume support: if a cached scores.json exists and --fresh is off, skip
    # folds whose booster .pt is already on disk. Survives OOM crashes mid-run
    # (just re-run the same command on a bigger machine).
    cached = {}
    if (not args.fresh) and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"  resume: {len(cached.get('scores', []))} fold(s) already done")
        except Exception as exc:
            print(f"  could not parse {cache_path}: {exc}; starting fresh")
            cached = {}
    if args.fresh:
        cached = {}
    if "target_std" in cached:
        target_std = cached["target_std"]
    done_scores = list(cached.get("scores", []))

    for fold_idx, (train_lf, valid_lf) in enumerate(folds):
        fname = f"booster_fold_{fold_idx}.pt"
        # Skip if this fold was completed and its weights are on disk.
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

        X_tr = train_df.select(raw_feature_cols).to_numpy().astype(np.float32)
        X_va = valid_df.select(raw_feature_cols).to_numpy().astype(np.float32)
        aid_tr = train_df["asset_id"].to_numpy().astype(np.int64)
        aid_va = valid_df["asset_id"].to_numpy().astype(np.int64)
        y_tr = train_df["target"].to_numpy().astype(np.float32)
        y_va = valid_df["target"].to_numpy().astype(np.float32)
        w_tr = train_df["weight"].to_numpy().astype(np.float32)
        w_va = valid_df["weight"].to_numpy().astype(np.float32)
        # NaN -> 0 for NN input (raw features have no NaN per memory, but guard).
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)

        if target_std is None:
            target_std = float(np.std(y_tr))

        # Tensors.
        Xtr_t = torch.tensor(X_tr, device=device)
        aidtr_t = torch.tensor(aid_tr, device=device)
        ytr_t = torch.tensor(y_tr, device=device)
        wtr_t = torch.tensor(w_tr, device=device)
        Xva_t = torch.tensor(X_va, device=device)
        aidva_t = torch.tensor(aid_va, device=device)
        yva_t = torch.tensor(y_va, device=device)
        wva_t = torch.tensor(w_va, device=device)

        model = DAEMLP(n_features=n_features, n_assets=n_assets,
                       asset_emb_dim=args.asset_emb_dim, hidden=args.hidden,
                       bottleneck=args.bottleneck, noise_std=args.noise_std).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        recon_loss_fn = nn.MSELoss()

        n = Xtr_t.shape[0]
        n_batches = math.ceil(n / args.batch_size)
        best_score = -1e9
        best_state = None
        best_epoch = -1
        patience, bad_epochs = 8, 0

        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            tot_recon = tot_tgt = 0.0
            for b in range(n_batches):
                idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
                xb = Xtr_t[idx]; aidb = aidtr_t[idx]; yb = ytr_t[idx]; wb = wtr_t[idx]
                opt.zero_grad()
                recon, pred, z = model(xb, aidb)
                l_recon = recon_loss_fn(recon, xb)
                # weighted target MSE
                l_tgt = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp(min=1e-8)
                loss = args.recon_weight * l_recon + args.target_weight * l_tgt
                loss.backward()
                opt.step()
                tot_recon += float(l_recon.detach()); tot_tgt += float(l_tgt.detach())
            # eval
            model.eval()
            with torch.no_grad():
                _, pred_va, _ = model(Xva_t, aidva_t)
                pred_va_np = pred_va.cpu().numpy()
            score = weighted_zero_mean_r2(y_va, pred_va_np, w_va)
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch}: recon={tot_recon/n_batches:.4f} tgt={tot_tgt/n_batches:.4f} "
                      f"valid_r2={score:.6f} (best {best_score:.6f}@{best_epoch})")
            if bad_epochs >= patience:
                print(f"  early stop at epoch {epoch}, best {best_score:.6f}@{best_epoch}")
                break

        # Save best model for this fold + incrementally update the resume cache.
        fname = f"booster_fold_{fold_idx}.pt"
        torch.save(best_state, out_dir / fname)
        fold_files.append(fname)
        scores.append(float(best_score))
        print(f"  fold {fold_idx} best valid R2: {best_score:.6f} -> {fname}")
        cache_path.write_text(json.dumps({"scores": scores, "target_std": target_std}, indent=2), encoding="utf-8")
        del Xtr_t, aidtr_t, ytr_t, wtr_t, Xva_t, aidva_t, yva_t, wva_t, model, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nCV scores: {[f'{s:.6f}' for s in scores]}")
    print(f"Mean CV: {float(np.mean(scores)):.6f}  Std: {float(np.std(scores)):.6f}")

    meta = {
        "backend": "dae_mlp",
        "model_type": "dae_mlp",
        "raw_feature_columns": list(raw_feature_cols),
        "feature_columns": list(raw_feature_cols),
        "n_features": n_features,
        "n_assets": n_assets,
        "asset_emb_dim": args.asset_emb_dim,
        "hidden": args.hidden,
        "bottleneck": args.bottleneck,
        "noise_std": args.noise_std,
        "n_folds": len(folds),
        "fold_files": fold_files,
        "target_std": target_std,
        "cv_mean": float(np.mean(scores)) if scores else None,
        "cv_std": float(np.std(scores)) if scores else None,
        "hparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "recon_weight": args.recon_weight, "target_weight": args.target_weight},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved meta -> {meta_path}")


if __name__ == "__main__":
    main()

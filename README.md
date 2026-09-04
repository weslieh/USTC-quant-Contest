# Quant Contest 2026 — Strategy Research Repository

This repository contains our research and strategy development for the 2026
quantitative trading research contest: predicting an anonymous risk-adjusted
`target` from ~300 anonymous features across 15 anonymous assets, evaluated by
**weighted zero-mean R²** under a strict time-series inference protocol.

## Problem & Constraints

- **Task**: predict `target` (risk-adjusted forward performance) per row;
  sign = direction, magnitude = confidence. Metric:
  `R² = 1 - Σ wᵢ(yᵢ-ŷᵢ)² / Σ wᵢ yᵢ²` — a *zero-mean* R² whose baseline is the
  all-zero prediction (score 0), not the mean.
- **Data**: train 13.2M rows / 9 parquet partitions / 15 assets / 323 features /
  47 responders; test 3.2M rows with features only. A strong multivariate
  distribution shift exists between train and test (adversarial AUC ≈ 1.0),
  while single-feature marginal drift is mild (median KS 0.06, none > 0.5).
- **Inference**: private-LB is a Time-Series API — `Model.predict(test)` is
  called once per `time_id` in strict ascending order, with **4 cores / 12 GB /
  no GPU / no network**. Per-step timeouts; failed steps are zeroed. Models must
  be loadable from a submitted ZIP with `main.py` at the root and must not use
  hardcoded absolute paths.

## Repository Layout

```
train.py                 GBDT trainer (LGB/XGB/Cat) with time-series CV
train_nn.py              Large supervised NN (periodic feature embeddings)
src/                     Feature engineering, CV, metrics, models, drift
strategy/                Self-contained single-model inference (main.py)
strategy_ensemble/       Multi-model weighted-ensemble inference
strategy_nn/             Self-contained NN inference (torch)
strategy_perasset*/      Per-asset independent-model inference
timeseries_api/          Local Time-Series API runner for validation
scripts/                 EDA, OOF generation, blending, scoring helpers
tests/                   Train/inference feature-parity tests
docs/                    Competition docs, handoff notes, baseline analysis
examples/                Official baseline strategies (LightGBM, linear, random)
```

## Branches & Research Lines

Work is organized across branches, each exploring a distinct hypothesis. The
chronological progression reflects an evolving understanding of what survives
the extreme distribution shift.

### `seed` — Multi-backend ensemble baseline
Three heterogeneous GBDT backends (LightGBM / XGBoost / CatBoost) trained on raw
features with weighted early stopping and expanding-window time CV. Established
the diversity ceiling: multi-seed bagging and OOF stacking each added only
~+0.00003 (essentially zero), since the three backends are highly correlated.
This is the raw-features + tree-baseline hard ceiling.

### `breakthrough` — `asset_id` as a categorical feature
The single most important structural discovery: `asset_id` was previously
ignored by all backends. Adding it as LightGBM `categorical_feature=[0]`
(optimal per-category splits; XGB/Cat treat it as numeric) was the *only*
"fixed per-row attribute" trick that survived the drift — because asset identity
is a permanent row property, not a train-derived structure. Established the key
rule: under AUC=1.0 drift, only fixed per-row properties survive; any
train-derived cross-sample/cross-time structure (cross-sectional, rolling,
reweighting, target transforms) fails.

### `asset-capacity` — Per-asset masked feature columns (best private-LB base)
Extended the asset-identity insight: for each asset's top-K features (by LGB
gain), add a column that holds the feature value on that asset's rows and 0
elsewhere (`pa_{asset}_{feature}`). Within-row and drift-safe. K=5 (→ 75 extra
columns, 399 total) was optimal; K=3 and K=8 were worse, and removing
"universal" features that recurred across assets was harmful. This single
shared model with three backends is the strongest configuration that runs within
the private-LB compute budget.

### `per-asset` — Fully independent per-asset models
A stronger but heavier form of asset modeling: 15 separate LGB/XGB/Cat models,
each trained on one asset's rows. Only +0.7% over the shared categorical model,
but 25× the inference time with timeouts — abandoned. Confirmed that a shared
tree with asset-aware splits beats fully independent per-asset models.

### `drift` / `adv` — Adversarial validation & drift handling
Explored adversarial-validation-based valid-set selection (pick train time_ids
most resembling test) and AV-based sample reweighting. AV-CV was useful as a
*lie detector* (it correctly flagged the target-rank transform's CV=0.77 as
fake), but AV-reweighting itself was harmful on the public LB (-14%): pushing
weights toward test-like samples ≈ training on the recent, low-signal tail.
Drop-drift feature filtering also hurt — no single feature truly drifts, so
filtering only discards signal.

### `responder-weight` — Responders as sample weights (dead)
Tried using |responder_03| (correlated 0.82 with target) and top-5 responders
to reweight training samples. All variants failed: absolute-value weighting
collapsed, top-5 went negative on full data. Same mechanism as AV-reweight —
train signal density ≠ test signal density under drift.

### `dae` — Denoising autoencoder (dead)
A small NN with reconstruction loss + supervised head, using LayerNorm (not
BatchNorm, to avoid cross-sample dependency). Scored negative on the public LB,
confirming that NN representations trained on raw features under AUC=1.0 drift
cannot extract signal GBDT misses. This was an *architecture* failure (small
network + reconstruction), not proof that NNs lack signal.

### `feature` / `eda` — Deep EDA & feature physics
Reverse-engineered feature physical types (price-level, return-like,
volatility-like, volume-event, ratio-probability) and ran deep EDA:
- Responders' covariance structure is extremely stable across early/mid/late
  train (Procrustes 0.9999), but responders are absent at inference — any
  method requiring responder *values* is dead.
- The strongest signals come from the *least*-drifting features (top-20 signal
  features have mean KS 0.031, half the population median).
- Lagged-correlation analysis confirmed the legitimate signal ceiling is ≈0.02
  (single feature |corr| with target); higher public-LB scores from other teams
  came from future-feature leakage (using the next time_id's feature), which is
  impossible under the private-LB's sequential release and against the rules.

### `feature-physics` — Large supervised NN + feature physics
A 945k-parameter NN with periodic (PLR-style) feature embeddings, an asset
embedding, a 4-layer residual MLP, and weighted-MSE supervision. This overturned the
"NN has no signal" conclusion from DAE: it scored positively and added
complementary signal to a GBDT ensemble (different mechanism → different errors).
However, the NN cannot run on the 4-core/no-GPU private-LB, so it is only
usable in the public-LB CSV phase. Diagnosed a "collapse to the mean" training
failure (output σ ≈ 0.04 vs target σ ≈ 1.09) — two loss-reform attempts (Pearson
correlation, then zero-referenced similarity) both regressed, revealing the
collapse was actually the *magnitude-optimal* solution for this 1:600 SNR
metric, not a bug. This branch also contains distillation scaffolding
(NN→GBDT soft-target injection) left unverified.

### `gbdt-baseline-learn` — Official-baseline-inspired causal history features
The most productive recent line, triggered by the official "no special tricks"
single-LightGBM baseline matching our 3-backend ensemble. The baseline's core
feature is **per-asset causal history** (`lag1` + `diff1` + `rmean5`, window=5,
no rolling std, top-48 by correlation) — which our memory had wrongly marked
dead (we had only tested cross-sectional *plus* rolling-with-std on a
pre-asset-categorical baseline). Re-tested cleanly:
- Window=3 beat window=5/10/2; fewer folds beat more (4-fold > 5-fold > 6-fold).
- Single-LGB with these features + per-asset columns + strong L2 (λ=50) is the
  strongest single model; the three-backend *ensemble* is harmful here because
  CatBoost drifts badly on history features (LB/CV ≈ 1.07 vs 1.55 for LGB/XGB).
- Includes `--extra-train-dir` support to append the 8/23 label backfill (the
  public-test period with labels, the closest training data to the private-LB
  live period), a local scoring script (zero-cost evaluation via the backfill
  labels), and a cross-path prediction-blend script.

## Methods Tried — Summary

**Survived (kept):** `asset_id` categorical; per-asset top-K masked feature
columns; per-asset causal history (lag1/diff1/rmean, small window, no std) on
single LGB with strong L2; weighted early stopping on the exact competition
metric; expanding-window time CV with embargo.

**Failed or neutral (not used):** cross-sectional features; rolling features
*with std*; feature interactions (redundant with what trees learn); drop-drift
filtering; target rank transform (CV fake, LB negative); AV sample reweighting;
multi-seed bagging (+0.00003); OOF stacking (+0.00003); per-asset fully
independent models (+0.7%, 25× slower); DAE (negative); responders as input or
weights (inference-unavailable or harmful); NN loss-reform to escape mean
collapse (the collapse is magnitude-optimal, not a bug); three-backend ensemble
*with* history features (CatBoost drift cancels the gain).

## Key Engineering Notes

- **Inference is self-contained**: `strategy*/main.py` cannot import `src/` (the
  submission package only puts the strategy dir on `sys.path`). Every transform
  is inlined. `tests/test_feature_parity.py` verifies train (Polars) and
  inference (NumPy) produce bit-identical features, one `time_id` slice at a time.
- **Thread capping**: `MAX_CPU_THREADS` (default 4) caps LGB/XGB/Cat/torch
  predict threads; leaving it unset causes cache thrashing and per-step time
  spikes that risk timeouts.
- **row_id semantics**: in private-LB, some rows only restore internal history
  and may have negative `row_id`; the inference code updates per-asset history
  for every row and never branches on `row_id`.
- **Polars 1.42 memory**: 323-column aggregations on 13.2M rows trigger false
  OOMs from expression-plan blowup; must chunk. `np.corrcoef` needs
  `rowvar=False`. scikit-learn must be pinned to 1.7.2 for xgboost compatibility.
- **Metric alignment**: GBDT early-stops on weighted zero-mean R² (LGB via a
  custom `feval`); XGB/Cat use built-in weighted RMSE since custom R² fevals
  degenerate their early stopping.

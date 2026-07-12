# Ensemble strategy

`strategy_ensemble/main.py` loads any number of independently-trained model
directories (LightGBM + XGBoost + CatBoost fold ensembles) and averages their
predictions weighted. Three backends give genuine diversity (CatBoost's
ordered boosting + symmetric trees differ from LGB/XGB), which is the main
lever left after same-backend bagging saturated.

## Setup

1. Train each backend (raw features only — cs/rolling were ablated as hurting
   the public LB). All MUST use the same feature spec (`--cs-topk 0
   --rolling-windows 0`) so the ensemble builds features once and feeds all.

   ```bash
   # LightGBM 5-fold (A config: num_leaves=64, min_child_samples=2000 — best on LB)
   python train.py --backend lgb --partitions 9 --n-folds 5 --embargo 5000 \
     --cs-topk 0 --rolling-windows 0 --num-leaves 64 --min-child-samples 2000 \
     --lr 0.03 --n-est 2000 --early-stopping-rounds 100 \
     --out-dir strategy_lgb --save-model

   # XGBoost 5-fold (max_depth=7, min_child_weight=4000 — strong reg, best on LB)
   python train.py --backend xgb --partitions 9 --n-folds 5 --embargo 5000 \
     --cs-topk 0 --rolling-windows 0 \
     --xgb-max-depth 7 --xgb-min-child-weight 4000 --lr 0.03 --n-est 2000 \
     --xgb-subsample 0.8 --xgb-colsample 0.8 --reg-alpha 0.1 --reg-lambda 1.0 \
     --early-stopping-rounds 100 --out-dir strategy_xgb --save-model

   # CatBoost 5-fold (depth=7; ordered boosting diversity)
   python train.py --backend cat --partitions 9 --n-folds 5 --embargo 5000 \
     --cs-topk 0 --rolling-windows 0 \
     --cat-depth 7 --cat-l2-leaf-reg 3.0 --lr 0.03 --n-est 2000 \
     --early-stopping-rounds 100 --out-dir strategy_cat --save-model
   ```

   Multi-seed bagging: re-run any backend with `--seed <n>` and a different
   `--out-dir` (e.g. `strategy_lgb_s999`), then add all dirs in step 2.

2. Generate `strategy_ensemble/ensemble_meta.json` listing all model dirs:

   ```bash
   python scripts/make_ensemble_meta.py \
     --model strategy_lgb:1 --model strategy_xgb:1 --model strategy_cat:1 \
     --out strategy_ensemble/ensemble_meta.json
   ```

   Each `--model` is `dir:weight` (weight optional, default 1). The script
   sanity-checks every meta (backend, n_folds, seed, cv_mean, feature spec)
   and prints a summary. Weights are normalised at inference time. Relative
   `dir` paths resolve against the ensemble dir's parent (project root).

   Tune weights by CV — start equal, then favour backends with higher CV
   (CatBoost and XGB tended slightly higher than LGB).

3. Run the local Time-Series API to produce a submission:

   ```bash
   python timeseries_api/run_timeseries_api.py \
     --data-root data --strategy-dir strategy_ensemble --output out/sub_ens.csv
   ```

## Inference time budget (important)

Eval env is 4 cores / 12 GB. Per-15-row time_id latency: LGB ~0.14ms,
XGB ~0.14ms, CatBoost ~1.35ms. For 214538 time_ids:

- 3 backends × 5 folds (15 boosters): ~29 min total — viable but watch the limit.
- 3 backends × 5 folds × 2 seeds (30 boosters): ~57 min — high timeout risk.

Always check `timing.total_seconds` and `timing.max_predict_seconds` in the
runner JSON before submitting. If too slow: drop CatBoost to 3 folds, lower
its ensemble weight, or drop multi-seed for CatBoost.

## Private-leaderboard packaging

Zip `strategy_ensemble/` together with every model dir it references
(`strategy_lgb/`, `strategy_xgb/`, `strategy_cat/`, ...). The ensemble main.py
reads them via relative paths, so keep the sibling layout under the project root.

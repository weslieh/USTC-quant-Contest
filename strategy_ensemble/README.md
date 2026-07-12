# Ensemble strategy

`strategy_ensemble/main.py` loads two independently-trained model directories
(LightGBM + XGBoost fold ensembles) and averages their predictions weighted.

## Setup

1. Train both backends (raw features only — cs/rolling were ablated as hurting
   the public LB). Both MUST use the same feature spec (same `--cs-topk 0
   --rolling-windows 0`), so the ensemble builds features once and feeds both.

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
   ```

2. Generate `strategy_ensemble/ensemble_meta.json` pointing at both dirs:

   ```bash
   python scripts/make_ensemble_meta.py \
     --lgb strategy_lgb --xgb strategy_xgb \
     --lgb-weight 0.5 --xgb-weight 0.5 \
     --out strategy_ensemble/ensemble_meta.json
   ```

   The script sanity-checks both metas (backend, n_folds, feature spec) and
   prints a summary. Tune the weights by CV — start 0.5/0.5, then favour
   whichever backend has higher CV (XGB tended slightly higher). Weights are
   normalised, so 0.5/0.5 == 50/50. Relative `dir` paths resolve against the
   ensemble dir's parent (project root) at inference time.

3. Run the local Time-Series API to produce a submission:

   ```bash
   python timeseries_api/run_timeseries_api.py \
     --data-root data --strategy-dir strategy_ensemble --output out/sub_ens.csv
   ```

For private-leaderboard submission, zip the `strategy_ensemble/` directory
together with both `strategy_lgb/` and `strategy_xgb/` model dirs (the ensemble
main.py reads them via relative paths, so keep the sibling layout).

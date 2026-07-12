# Ensemble strategy

`strategy_ensemble/main.py` loads two independently-trained model directories
(LightGBM + XGBoost fold ensembles) and averages their predictions weighted.

## Setup

1. Train both backends (raw features only — cs/rolling were ablated as hurting
   the public LB). Both MUST use the same feature spec (same `--cs-topk 0
   --rolling-windows 0`), so the ensemble builds features once and feeds both.

   ```bash
   # LightGBM 5-fold
   python train.py --backend lgb --partitions 9 --n-folds 5 --embargo 5000 \
     --cs-topk 0 --rolling-windows 0 --num-leaves <nl> --min-child-samples <mcs> \
     --out-dir strategy_lgb --save-model

   # XGBoost 5-fold
   python train.py --backend xgb --partitions 9 --n-folds 5 --embargo 5000 \
     --cs-topk 0 --rolling-windows 0 \
     --xgb-max-depth 6 --xgb-min-child-weight 5 --lr 0.03 --n-est 2000 \
     --out-dir strategy_xgb --save-model
   ```

2. Create `strategy_ensemble/ensemble_meta.json` pointing at both dirs with
   weights (relative paths resolve against the ensemble dir's parent, i.e. the
   project root):

   ```json
   {"models": [
     {"dir": "strategy_lgb", "weight": 0.6},
     {"dir": "strategy_xgb", "weight": 0.4}
   ]}
   ```

   Tune the weights by CV — start 0.5/0.5, then favour whichever backend has
   higher last-fold CV. The weights are normalised, so 0.6/0.4 == 60/40.

3. Run the local Time-Series API to produce a submission:

   ```bash
   python timeseries_api/run_timeseries_api.py \
     --data-root data --strategy-dir strategy_ensemble --output out/sub_ens.csv
   ```

For private-leaderboard submission, zip the `strategy_ensemble/` directory
together with both `strategy_lgb/` and `strategy_xgb/` model dirs (the ensemble
main.py reads them via relative paths, so keep the sibling layout).

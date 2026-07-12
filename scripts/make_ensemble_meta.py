"""Generate strategy_ensemble/ensemble_meta.json from two trained model dirs.

Usage:
    python scripts/make_ensemble_meta.py --lgb strategy_lgb --xgb strategy_xgb \
        --lgb-weight 0.5 --xgb-weight 0.5 --out strategy_ensemble/ensemble_meta.json

Both sub-models must share the same feature spec (raw-only recommended).
Relative paths in the output resolve against the ensemble dir's parent at
inference time, so prefer relative paths when the dirs sit beside strategy_ensemble.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lgb", required=True, help="LightGBM strategy dir (has model_meta.json).")
    p.add_argument("--xgb", required=True, help="XGBoost strategy dir (has model_meta.json).")
    p.add_argument("--lgb-weight", type=float, default=0.5)
    p.add_argument("--xgb-weight", type=float, default=0.5)
    p.add_argument("--out", default="strategy_ensemble/ensemble_meta.json")
    args = p.parse_args()

    # Sanity-check both metas exist and agree on engineered-feature spec.
    for d in (args.lgb, args.xgb):
        meta = json.loads((Path(d) / "model_meta.json").read_text(encoding="utf-8"))
        print(f"{d}: backend={meta.get('backend')} n_folds={meta.get('n_folds')} "
              f"cs={len(meta.get('cs_source_columns', []))} "
              f"roll={len(meta.get('rolling_source_columns', []))} "
              f"cv_mean={meta.get('cv_mean')}")

    lgb_meta = json.loads((Path(args.lgb) / "model_meta.json").read_text(encoding="utf-8"))
    xgb_meta = json.loads((Path(args.xgb) / "model_meta.json").read_text(encoding="utf-8"))
    if lgb_meta.get("raw_feature_columns") != xgb_meta.get("raw_feature_columns"):
        print("WARNING: raw feature columns differ between LGB and XGB — "
              "ensemble requires identical feature spec.")

    out = {
        "models": [
            {"dir": args.lgb, "weight": args.lgb_weight},
            {"dir": args.xgb, "weight": args.xgb_weight},
        ]
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

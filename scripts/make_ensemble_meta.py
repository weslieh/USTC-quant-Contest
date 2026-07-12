"""Generate strategy_ensemble/ensemble_meta.json from trained model dirs.

Supports any number of sub-models (multi-seed bagging, LGB+XGB, etc.).

Usage (two-model LGB+XGB):
    python scripts/make_ensemble_meta.py \
        --model strategy_lgb:0.45 --model strategy_xgb:0.55 \
        --out strategy_ensemble/ensemble_meta.json

Usage (multi-seed bagging, 4 models equal weight):
    python scripts/make_ensemble_meta.py \
        --model strategy_lgb_s42:1 --model strategy_lgb_s123:1 \
        --model strategy_xgb_s42:1 --model strategy_xgb_s123:1 \
        --out strategy_ensemble/ensemble_meta.json

Each --model is "dir:weight" (weight optional, defaults to 1). All sub-models
must share the same engineered-feature spec (raw-only recommended) so features
are built once and fed to all. Weights are normalised at inference time.
Relative dir paths resolve against the ensemble dir's parent (project root).
"""
import argparse
import json
from pathlib import Path


def parse_model(s):
    if ":" in s:
        d, w = s.rsplit(":", 1)
        return d.strip(), float(w)
    return s.strip(), 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True,
                   help='"dir:weight" (weight optional, default 1). Repeatable.')
    p.add_argument("--out", default="strategy_ensemble/ensemble_meta.json")
    args = p.parse_args()

    models = []
    for spec in args.model:
        d, w = parse_model(spec)
        meta_path = Path(d) / "model_meta.json"
        if not meta_path.exists():
            raise SystemExit(f"missing {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"{d}: backend={meta.get('backend')} n_folds={meta.get('n_folds')} "
              f"seed={meta.get('hparams',{}).get('random_state')} "
              f"cv_mean={meta.get('cv_mean')} weight={w}")
        models.append({"dir": d, "weight": w})

    # Sanity-check feature spec agreement across all models.
    base_raw = json.loads((Path(models[0]["dir"]) / "model_meta.json")
                          .read_text(encoding="utf-8")).get("raw_feature_columns")
    for m in models[1:]:
        raw = json.loads((Path(m["dir"]) / "model_meta.json")
                         .read_text(encoding="utf-8")).get("raw_feature_columns")
        if raw != base_raw:
            print(f"WARNING: raw feature columns differ between {models[0]['dir']} and {m['dir']}")

    out = {"models": models}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    total = sum(m["weight"] for m in models)
    print(f"Wrote {args.out} — {len(models)} models, total weight {total}")


if __name__ == "__main__":
    main()

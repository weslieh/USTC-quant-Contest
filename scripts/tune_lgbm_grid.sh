#!/usr/bin/env bash
# LightGBM hyperparameter grid search on a small sample.
# Raw features only (cs/rolling ablated — they hurt the public leaderboard).
#
# Usage (cloud):
#   bash scripts/tune_lgbm_grid.sh
#   # override knobs via env:
#   PARTS=3 FOLDS=3 EMBARGO=5000 DATA_ROOT=data OUT_BASE=/tmp/tune bash scripts/tune_lgbm_grid.sh
#
# Each run writes its CV scores to <out>/cv/scores.json. At the end we print a
# summary table: mean / std / last-fold (fold5 ≈ most recent regime, closest to
# test distribution). Pick by last-fold primarily, mean secondarily — CV and
# public LB diverge, so the most-recent fold is the more trustworthy signal.
set -u
DATA_ROOT="${DATA_ROOT:-data}"
OUT_BASE="${OUT_BASE:-/tmp/tune}"
PARTS="${PARTS:-3}"
FOLDS="${FOLDS:-3}"
EMBARGO="${EMBARGO:-5000}"
LR="${LR:-0.03}"
N_EST="${N_EST:-2000}"
FF="${FF:-0.8}"
BF="${BF:-0.8}"
BFQ="${BFQ:-1}"
RALPHA="${RALPHA:-0.1}"
RLAMBDA="${RLAMBDA:-0.1}"
ES="${ES:-100}"

mkdir -p "$OUT_BASE"
SUMMARY="$OUT_BASE/summary.tsv"
echo -e "tag\tmean\tstd\tlast_fold\tnum_leaves\tmin_child" > "$SUMMARY"

run() {
  local tag="$1" nl="$2" mcs="$3"
  local outdir="$OUT_BASE/$tag"
  echo "=== $tag (num_leaves=$nl min_child=$mcs) ==="
  python train.py \
    --data-root "$DATA_ROOT" --partitions "$PARTS" --n-folds "$FOLDS" \
    --valid-frac 0.1 --embargo "$EMBARGO" \
    --cs-topk 0 --rolling-windows 0 \
    --num-leaves "$nl" --min-child-samples "$mcs" \
    --lr "$LR" --n-est "$N_EST" --feature-frac "$FF" --bagging-frac "$BF" \
    --bagging-freq "$BFQ" --reg-alpha "$RALPHA" --reg-lambda "$RLAMBDA" \
    --early-stopping-rounds "$ES" \
    --out-dir "$outdir" --fresh 2>&1 | grep -E "Mean CV" | tail -1
  # parse scores.json -> mean / std / last
  python - "$outdir/cv/scores.json" "$tag" "$nl" "$mcs" "$SUMMARY" <<'PY'
import json, sys, statistics
sp, tag, nl, mcs, out = sys.argv[1:6]
scores = json.loads(open(sp).read())["scores"]
mean = statistics.mean(scores)
std = statistics.pstdev(scores)
last = scores[-1]
with open(out, "a") as f:
    f.write(f"{tag}\t{mean:.6f}\t{std:.6f}\t{last:.6f}\t{nl}\t{mcs}\n")
print(f"  -> mean={mean:.6f} std={std:.6f} last_fold={last:.6f}")
PY
}

# Highest-leverage grid for weak-signal time series: min_child_samples x num_leaves.
for mcs in 500 1000 2000; do
  for nl in 31 63 127; do
    run "mcs${mcs}_nl${nl}" "$nl" "$mcs"
  done
done

echo
echo "================ GRID SUMMARY (sort by last_fold desc) ================"
sort -t$'\t' -k4 -rn "$SUMMARY" | column -t -s$'\t'
echo
echo "Pick the top row (highest last_fold); verify mean isn't much worse, then"
echo "full-train with --partitions 9 --n-folds 5 and submit to public LB."

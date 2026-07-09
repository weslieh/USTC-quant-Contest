#!/usr/bin/env bash
# Entrypoint for the training container.
#
# Data comes in via a mounted volume at $DATA_ROOT (default /mnt/data).
# Artifacts go out to $OUT_DIR (default /mnt/output). On managed platforms
# these are mounted for you; locally you just `docker run -v ...`.
#
# Optional object-storage sync (provider-agnostic, off by default):
#   INPUT_URI  - if set, `gsutil`/`aws s3`/`ossutil` is auto-selected by scheme
#                and used to pull data into $DATA_ROOT before training.
#   OUTPUT_URI - if set, artifacts are pushed back after training.
#   GSUTIL_* / AWS_* / OSS_* env vars carry credentials (mount as secrets).
#
# Training flags: pass them via TRAIN_ARGS env var, or as command args which
# override TRAIN_ARGS entirely.

set -euo pipefail

echo "=== train-entrypoint ==="
echo "DATA_ROOT=$DATA_ROOT  OUT_DIR=$OUT_DIR  TRAIN_ARGS=$TRAIN_ARGS"

mkdir -p "$DATA_ROOT" "$OUT_DIR"

# ---- Optional: pull data from object storage before training ------------
if [[ -n "${INPUT_URI:-}" ]]; then
    echo ">>> syncing input data from $INPUT_URI"
    case "$INPUT_URI" in
        gs://*)   gsutil -m cp -r "$INPUT_URI"/* "$DATA_ROOT"/ ;;
        s3://*)   aws s3 cp --recursive "$INPUT_URI" "$DATA_ROOT" ;;
        oss://*)  ossutil cp -rf "$INPUT_URI" "$DATA_ROOT"/ ;;
        *)
            # Generic fallback: try rclone if a remote is configured.
            if command -v rclone >/dev/null 2>&1; then
                rclone copy "$INPUT_URI" "$DATA_ROOT" --progress
            else
                echo "unsupported INPUT_URI scheme: $INPUT_URI" >&2; exit 1
            fi ;;
    esac
fi

# ---- Sanity: data must be present ---------------------------------------
if [[ ! -f "$DATA_ROOT/manifest.json" ]]; then
    echo "ERROR: $DATA_ROOT/manifest.json not found." >&2
    echo "Mount data at \$DATA_ROOT or set INPUT_URI to sync it." >&2
    exit 1
fi

# ---- Determine train arguments ------------------------------------------
# Direct command args (CMD) override TRAIN_ARGS; else use TRAIN_ARGS.
if [[ "$#" -gt 0 ]]; then
    ARGS=("$@")
else
    # shellcheck disable=SC2206
    ARGS=($TRAIN_ARGS)
fi

# Default to a sensible full run if nothing was specified.
if [[ "${#ARGS[@]}" -eq 0 ]]; then
    ARGS=(--data-root "$DATA_ROOT" --out-dir "$OUT_DIR" --save-model)
elif [[ " ${ARGS[*]} " != *" --data-root "* ]]; then
    ARGS+=(--data-root "$DATA_ROOT")
fi
if [[ " ${ARGS[*]} " != *" --out-dir "* ]]; then
    ARGS+=(--out-dir "$OUT_DIR")
fi

echo ">>> running: python train.py ${ARGS[*]}"
cd /workspace
exec python train.py "${ARGS[@]}"

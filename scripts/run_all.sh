#!/usr/bin/env bash
# Reproduce the whole project from scratch:
#   download -> preprocess -> train 3 models -> evaluate + figures.
# Usage:  bash scripts/run_all.sh [image_size]     (default 64)
set -euo pipefail
SIZE="${1:-64}"
PY="${PYTHON:-python}"
mkdir -p runs

"$PY" scripts/download_data.py
"$PY" scripts/preprocess.py

for MODEL in ae vae fanogan; do
  "$PY" scripts/train.py --config "configs/${MODEL}.yaml" --set "data.image_size=${SIZE}" \
    2>&1 | tee "runs/train_${MODEL}_${SIZE}.log"
done

"$PY" scripts/evaluate.py --baseline \
  --runs "runs/ae_${SIZE}" "runs/vae_${SIZE}" "runs/fanogan_${SIZE}"

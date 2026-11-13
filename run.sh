#!/bin/bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

echo "[INFO] Starting evaluation..."
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python3 main_eval.py \
  --tasks-json ./submit/150_ko_metadata.json \
  --results-root ./results \
  --max-images 0 \
  --image-chunk-size 8 \
  --port 8005 \
  --workers 64 \
  --request-timeout 120

echo "[INFO] Evaluation finished."

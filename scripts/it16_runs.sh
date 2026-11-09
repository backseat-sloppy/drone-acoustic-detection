#!/usr/bin/env bash
# Iteration 16 (report entry 155): every arm for one seed on one GPU.
#   nohup bash scripts/it16_runs.sh SEED GPU > artifacts/runs/it16_s$SEED.log 2>&1 &
# On the server call .venv/bin/python, never uv run (entry 091).
set -euo pipefail
seed=$1; gpu=$2
cd "$(dirname "$0")/.."
run() {  # run NAME [extra run_range.py args...]
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u scripts/run_range.py --name "it16_$1_s$seed" \
    --pool --arms none --recipes simdist --device cuda --speed 0.8 1.25 --deploy-fa 0.05 \
    --seed "$seed" --frontend mfe "${@:2}"
}
base=(--models dscnn crnn)
[ "$seed" = 0 ] && base=(--models comb lgbm dscnn crnn)   # baselines once, in the reference
run ref    "${base[@]}"
run 600k   --models dscnn crnn --windows 600000
run win2s  --models dscnn crnn --window-seconds 2.0
run wide   --models dscnnwide crnnwide
run mel128 --models dscnn crnn --n-mels 128 --n-fft 800

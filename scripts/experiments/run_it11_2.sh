#!/usr/bin/env bash
# Iteration 11.2 (entry 143): real drones from several rigs, leave one rig out, CRNN against the frozen
# Cnn14_16k probe, on it9's pooled recipe without FPV. Runs in Marius's server account.
#
#   bash scripts/experiments/run_it11_2.sh check     # data and checkpoint present, interpreter OK
#   bash scripts/experiments/run_it11_2.sh train     # three background jobs on GPUs 0-2 (see below)
#   bash scripts/experiments/run_it11_2.sh status    # last log line of each job
#   bash scripts/experiments/run_it11_2.sh gates     # gates (a)-(e) for every trained model, it9 alongside
#
# train: run_range.py --rule143, recipes simdist (it9) and simdist_steady (+ steady noise), speed 0.8-1.25,
# 200 000 windows per CRNN (as it9), 100 000 per probe, threshold at 5% FA on the training rig's validation.
#   GPU 0: seed 0, Nightjar held out: comb, lgbm, crnn, panns_probe
#   GPU 2: seed 0, d3set held out:    comb, lgbm, crnn, panns_probe
#   GPU 1: seed 1, both folds:        crnn
# Runs .venv/bin/python, else ~/dad/.venv; never `uv run` (Pascal GPUs, see run_it11_1.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=$HOME/dad/.venv/bin/python; fi
fi
OUT=artifacts/it11_2
mkdir -p "$OUT/logs"
stamp() { date +%Y%m%d_%H%M%S; }
COMMON=(--rule143 --recipes simdist simdist_steady --speed 0.8 1.25 --windows 200000 --probe-windows 100000
        --deploy-fa 0.05 --eval-hop 0.5 --device cuda)

case "${1:-}" in
check)
  "$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.device_count())"
  for f in manifest.csv range_track.csv nightjar.i16 d3set.i16 ddl.f32 fsd50k.i16; do
    ls -l "data/processed/$f" || { echo "MISSING data/processed/$f"; exit 1; }
  done
  ls -l "artifacts/pretrained/Cnn14_16k_mAP=0.438.pth"
  "$PY" -c "
import sys; sys.path.insert(0, 'src')
from dad import rangerun as rg; from dad.realdata import load_manifest
m = load_manifest()
for held in rg.POOL143_UNITS:
    a = rg.make_pool_arm_143(m, held)
    print(f'hold out {held}: train {len(a.train)} segments, val {len(a.val)} (from {sorted({r.source for r in a.val})}), '
          f'test {len(a.test)}, mower {len(a.confuser)}; train sources {sorted({r.source for r in a.train})}')
# Build each fold's sampler as train does (this reads the audio, so a missing source fails here, not on the GPUs)
from dad import realrun as rr; from dad.config import FrontendConfig
cfg = FrontendConfig(); splits = rr.make_splits(m, cfg)
for held in rg.POOL143_UNITS:
    for rec in ('simdist', 'simdist_steady'):
        sm = rg.RangeSampler(rg.make_pool_arm_143(m, held), splits, rg.RECIPES[rec], cfg, seed=0, speed=(0.8, 1.25), dronenoise=False)
        assert not [c for c in sm.cells if c.chain == 'dronenoise'], 'DroneNoise in a rule-143 sampler'
        print(f'hold out {held}, {rec}:', sm.describe())
"
  ;;
train)
  t=$(stamp)
  launch() {  # gpu name args...
    local g=$1 n=$2; shift 2
    CUDA_VISIBLE_DEVICES=$g nohup setsid "$PY" -u scripts/run_range.py --name "$n" "$@" "${COMMON[@]}" \
        > "$OUT/logs/${n}_$t.log" 2>&1 < /dev/null &
    echo "GPU $g: $n (pid $!), log $OUT/logs/${n}_$t.log"
  }
  launch 0 it11_2_s0_nightjar --arms nightjar --models comb lgbm crnn panns_probe --seed 0
  launch 2 it11_2_s0_d3set    --arms d3set    --models comb lgbm crnn panns_probe --seed 0
  launch 1 it11_2_s1          --arms nightjar d3set --models crnn --seed 1
  echo "Running in the background. Progress: bash scripts/experiments/run_it11_2.sh status"
  ;;
status)
  for f in "$OUT"/logs/it11_2_*.log; do echo "== $f"; tail -n 2 "$f"; done
  pgrep -af "run_range.py --name it11_2" | sed 's/--rule143.*//' || echo "(no it11_2 job running)"
  ;;
gates)
  pgrep -f "run_range.py --name it11_2" >/dev/null && { echo "training still running; wait for it"; exit 1; }
  IT9="it9 CRNN=artifacts/models/final_crnn_none_simdist_crnn.pt"
  mapfile -t M  < <("$PY" scripts/it11_2_models.py --no-threshold)
  mapfile -t MT < <("$PY" scripts/it11_2_models.py)
  [ ${#M[@]} -gt 0 ] || { echo "no trained it11_2 checkpoints found"; exit 1; }
  printf '%s\n' "${MT[@]}" | tee "$OUT/models.txt"
  G=${GPU:-0}
  # (b), (c): each fold's model on the held-out rig's validation sessions (the rows that count: the held rig).
  CUDA_VISIBLE_DEVICES=$G "$PY" -u scripts/score_real_dev.py --device cuda --out "$OUT/gates_real_dev.json" "$IT9" "${M[@]}" \
      2>&1 | tee "$OUT/logs/gates_real_dev_$(stamp).log"
  # (e) at each model's fold threshold; it9 at its own.
  CUDA_VISIBLE_DEVICES=$G "$PY" -u scripts/score_mower.py --device cuda --out "$OUT/gates_mower.json" "$IT9@0.4827928543090821" "${MT[@]}" \
      2>&1 | tee "$OUT/logs/gates_mower_$(stamp).log"
  # (a), (d)
  CUDA_VISIBLE_DEVICES=$G "$PY" -u scripts/score_glasgow.py --device cuda --room-second-half --war-snr 0 -5 -10 \
      --out "$OUT/gates_glasgow.json" "$IT9@0.4827928543090821" "${MT[@]}" 2>&1 | tee "$OUT/logs/gates_glasgow_$(stamp).log"
  ;;
*)
  sed -n '2,16p' "$0"; exit 1 ;;
esac

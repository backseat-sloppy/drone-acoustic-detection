#!/usr/bin/env bash
# Branch 11.1: the PANNs Cnn14_16k arm (plan of 2026-11-06), on the JupyterLab server.
#
#   bash scripts/experiments/run_it11_1.sh fetch     # checkpoint + AudioSet labels, MD5s printed and saved
#   bash scripts/experiments/run_it11_1.sh test      # tests/test_panns.py, incl. the checkpoint checks
#   bash scripts/experiments/run_it11_1.sh smoke     # tiny P1 on 20 scenes into artifacts/it11_1/smoke (paths, GPU)
#   bash scripts/experiments/run_it11_1.sh p0        # AudioSet scores, no training (1 GPU, minutes)
#   bash scripts/experiments/run_it11_1.sh p1        # frozen encoder + probe (1 GPU)
#   bash scripts/experiments/run_it11_1.sh p2        # LP-FT, seeds 0-2 on GPUs 0-2, in the background
#   bash scripts/experiments/run_it11_1.sh gates     # real-rig and Glasgow gates, where the data exists here
#   bash scripts/experiments/run_it11_1.sh pack      # one tar of artifacts/it11_1 (no checkpoints over 100 MB) + MD5
#
# Every stage writes under artifacts/it11_1/ and logs to artifacts/it11_1/logs/. The interpreter is
# $PY (default: the repo's .venv/bin/python), the one the CRNN runs already train with; never `uv run`:
# uv may re-sync torch, and recent CUDA builds of torch dropped the Pascal GPUs (TITAN X) this server
# has. fp32 throughout (Pascal fp16 is slow). Example: PY=$(which python) bash scripts/experiments/run_it11_1.sh test
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
OUT=artifacts/it11_1
PRE=artifacts/pretrained
CK="$PRE/Cnn14_16k_mAP=0.438.pth"
SCENES=${SCENES:-data/processed/scenes}
NODRONE=${NODRONE:-$HOME/nodrone}
# The e133 compliant data (entries 133-134), so the network is the only change:
DATA=(--drones "$SCENES/glasgow_d1_16" "$SCENES/it11s_valdrone" "$SCENES/it11s/nodrone"
      --nodrone "$NODRONE" --speed 0.8 1.25 --eq --mix-real 0.8)
mkdir -p "$OUT/logs" "$PRE"
stamp() { date +%Y%m%d_%H%M%S; }
note() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/logs/run.log"; }

preflight() {   # the interpreter must exist and have torch, CUDA and the repo's other dependencies
  command -v "$PY" >/dev/null 2>&1 || { echo "No interpreter at PY=$PY. Set PY to the python the CRNN runs use, e.g. PY=\$(which python)"; exit 1; }
  "$PY" -c "import torch, numpy, scipy, sklearn, soundfile, lightgbm; print('python', __import__('sys').executable, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.device_count())" \
    | tee -a "$OUT/logs/run.log"
}

case "${1:-}" in
fetch)
  if [ ! -f "$CK" ]; then
    curl -fL --retry 3 -o "$CK.part" "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
    mv "$CK.part" "$CK"
  fi
  [ -f "$PRE/class_labels_indices.csv" ] || curl -fL --retry 3 -o "$PRE/class_labels_indices.csv" \
      "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
  (cd "$PRE" && md5sum "Cnn14_16k_mAP=0.438.pth" class_labels_indices.csv | tee MD5SUMS)
  ls -l "$PRE"
  note "fetch done: $(cat $PRE/MD5SUMS | tr '\n' ' ')"
  ;;
test)
  preflight
  $PY -m pytest -q -p no:cacheprovider tests/test_panns.py 2>&1 | tee "$OUT/logs/test_$(stamp).log"
  ;;
smoke)
  preflight
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u scripts/train_panns.py --mode probe --name smoke --limit 20 --windows 512 \
      --device cuda --out "$OUT/smoke" "${DATA[@]}" 2>&1 | tee "$OUT/logs/smoke_$(stamp).log"
  ;;
p0)
  preflight
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u scripts/panns_p0.py --device cuda --out "$OUT/p0.json" \
      --drones "$SCENES/it11s_valdrone" --nodrone "$NODRONE" 2>&1 | tee "$OUT/logs/p0_$(stamp).log"
  ;;
p1)
  preflight
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -u scripts/train_panns.py --mode probe --name p1_s0 --seed 0 \
      --windows ${WINDOWS:-100000} --device cuda --out "$OUT" "${DATA[@]}" 2>&1 | tee "$OUT/logs/p1_$(stamp).log"
  ;;
p2)
  preflight
  [ -f "$OUT/p1_s0_panns.pt" ] || { echo "run p1 first (LP-FT starts from its probe)"; exit 1; }
  for s in 0 1 2; do
    log="$OUT/logs/p2_s${s}_$(stamp).log"
    CUDA_VISIBLE_DEVICES=$s nohup $PY -u scripts/train_panns.py --mode ft --init "$OUT/p1_s0_panns.pt" \
        --name "p2_s$s" --seed $s --windows ${WINDOWS:-600000} --device cuda --out "$OUT" "${DATA[@]}" \
        > "$log" 2>&1 &
    note "p2 seed $s on GPU $s, pid $!, log $log"
  done
  echo "Running in the background; follow with: tail -f $OUT/logs/p2_s*_*.log"
  ;;
gates)
  preflight
  models=()
  for f in "$OUT"/p1_s0_panns.pt "$OUT"/p2_s*_panns.pt; do [ -f "$f" ] && models+=("$(basename "$f" _panns.pt)=$f"); done
  [ ${#models[@]} -gt 0 ] || { echo "no checkpoints in $OUT"; exit 1; }
  if [ -f data/processed/manifest.csv ]; then
    $PY -u scripts/score_real_dev.py --device cuda --out "$OUT/gates_real_dev.json" "${models[@]}" \
        2>&1 | tee "$OUT/logs/gates_real_dev_$(stamp).log"
  else
    note "gates: no data/processed/manifest.csv in this account; real-rig gates (b, c) and the mower gate must run where it is"
  fi
  if [ -d data/extracted/Glasgow ]; then
    # Frozen threshold: each run's own 5%-FA threshold on train_clean's validation (synthetic drones vs the
    # Nightjar no-drone clips). Entry 134 used the real-background threshold (it9's rule, real_threshold.py,
    # Mads's account); score with that too before any gate verdict.
    thr=()
    for m in "${models[@]}"; do
      n=${m%%=*}; t=$($PY -c "import json,sys; d=json.load(open('$OUT/$n.json')); print([v for k,v in d['models'].items() if k.startswith('panns')][0]['threshold_5pct_fa'])")
      thr+=("$m@$t")
    done
    $PY -u scripts/score_glasgow.py --device cuda --room-second-half --war-snr 0 -5 -10 \
        --out "$OUT/gates_glasgow.json" "${thr[@]}" 2>&1 | tee "$OUT/logs/gates_glasgow_$(stamp).log"
  else
    note "gates: no data/extracted/Glasgow in this account; gates (a, d) must run where it is"
  fi
  ;;
pack)
  tarf="it11_1_results_$(stamp).tar"
  find "$OUT" -type f -size -100M | tar -cf "$tarf" -T -
  md5sum "$tarf" | tee "$tarf.md5"
  ls -l "$tarf"; echo "Fine-tuned checkpoints (about 320 MB each) are left out; list:"; ls -l "$OUT"/*_panns.pt 2>/dev/null || true
  ;;
*)
  sed -n '2,15p' "$0"; exit 1 ;;
esac

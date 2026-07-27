#!/bin/bash
# CIFAR-10 pilot sweep (docs/THINKING/G3-cifar-pilot.md, prereg QG-4..QG-8).
# Fits a 2000-image val subset under P-shared-det at each candidate config, then runs the
# task-referenced quality gate with the strengthened 10-epoch reference CNN.
# Freeze rule: cheapest config with acc_real - acc_render <= 1.0 pt.
#
# Detached usage (survives machine sleep):
#   nohup caffeinate -i bash scripts/09_cifar_pilot.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/cifar_pilot.log
mkdir -p results

# config id : width layers steps
CONFIGS=(
  "A 32 2 1000"
  "B 64 3 500"
  "C 64 3 1000"
  "D 64 3 2000"
)

{
  echo "=== cifar pilot sweep start $(date) ==="
  for cfg in "${CONFIGS[@]}"; do
    read -r id w l s <<< "$cfg"
    tag="pilot-w${w}L${l}s${s}"
    echo "--- config $id: width=$w layers=$l steps=$s (tag $tag) $(date) ---"
    $PY scripts/03_generate_inrbench.py --dataset cifar10 --protocol P-shared-det \
        --split val --limit 2000 --steps "$s" --width "$w" --layers "$l" \
        --batch 256 --tag "$tag" || { echo "GEN FAILED $id"; continue; }
    $PY scripts/04_quality_gate.py --dir "data/inrbench/cifar10/P-shared-det-${tag}" \
        --dataset cifar10 --eval-split val --gate-epochs 10 || echo "GATE FAILED $id"
    echo "--- config $id done $(date) ---"
  done
  echo "=== cifar pilot sweep complete $(date) ==="
} >> "$LOG" 2>&1

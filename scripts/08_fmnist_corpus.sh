#!/bin/bash
# FMNIST sine corpus at frozen config (w32 L2 steps300; pilot gate passed 2026-07-18).
# Waits for any running generation job to finish first, then runs the 4-protocol chain.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
{
  echo "=== fmnist chain queued $(date) ==="
  while pgrep -f "03_generate_inrbench" > /dev/null; do sleep 60; done
  echo "=== fmnist chain start $(date) ==="
  GEN="scripts/03_generate_inrbench.py --dataset fashionmnist --steps 300 --width 32 --layers 2 --batch 256"
  $PY $GEN --protocol P-shared-det   --split all   && echo "FMNIST DET DONE $(date)"
  $PY $GEN --protocol P-random       --split all   && echo "FMNIST RANDOM DONE $(date)"
  $PY $GEN --protocol P-shared-stoch --split all   && echo "FMNIST STOCH DONE $(date)"
  $PY $GEN --protocol P-random-K     --split train && echo "FMNIST K DONE $(date)"
  echo "=== fmnist chain complete $(date) ==="
} >> results/night_chain.log 2>&1

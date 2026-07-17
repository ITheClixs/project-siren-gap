#!/bin/bash
# Detached night chain: anchors (A1/A2) then P-random-K corpus. Shard-resumable;
# safe to rerun. Progress -> results/night_chain.log; done-markers per stage.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
{
  echo "=== night chain start $(date) ==="
  if [ ! -f results/anchors/anchors_mnist.json ]; then
    $PY scripts/06_anchors.py --seeds 5 && echo "ANCHORS DONE $(date)"
  else
    echo "anchors already done"
  fi
  $PY scripts/03_generate_inrbench.py --dataset mnist --protocol P-random-K --split train \
     --steps 300 --width 32 --layers 2 --batch 256 && echo "RANDOM-K DONE $(date)"
  echo "=== night chain complete $(date) ==="
} >> results/night_chain.log 2>&1

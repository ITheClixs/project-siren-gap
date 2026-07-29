#!/bin/bash
# G5 stage 1: the S1 decomposition ladder on CIFAR-10, against the frozen registration
# docs/prereg/S1-cifar.md (sha256-16 f7906fc6904c7c81), then the confirmatory analysis.
#
# Third dataset arm: natural RGB images, c = 3 output channels, a different render-fidelity
# regime (40.1 dB, render penalty at the gate noise floor). Absolute levels are not expected
# to transfer from MNIST/FMNIST; the recovery fractions are what is registered.
#
# Detached usage (survives lid close / sleep):
#   nohup caffeinate -i bash scripts/20_cifar_ladder.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/cifar_ladder.log
mkdir -p results

# Match the python process, never a bare -f pattern: `pgrep -f <literal>` also matches any shell
# whose command line contains it, including the polling loop itself (deadlock of 2026-07-28).
gen_running() { pgrep -fl "03_generate_inrbench" 2>/dev/null | grep -q "[.]venv/bin/python"; }

# cheapest and most central rungs first, then the augmentation-bearing ones (15 seeds),
# then W7 (8x training rows, heaviest in memory)
RUNGS="P0 P1 W1 W2 W3 W4 W5 W10 W9 X1 W6 W8 W7-1/8 W7"

{
  echo "=== cifar ladder start $(date) ==="
  while gen_running; do sleep 60; done

  for r in $RUNGS; do
    echo "--- rung $r $(date) ---"
    $PY scripts/11_ladder.py --dataset cifar10 --rungs "$r" || echo "RUNG FAILED $r"
  done

  echo "--- analysis $(date) ---"
  $PY scripts/14_ladder_analysis.py --dataset cifar10 || echo "ANALYSIS FAILED"
  echo "=== cifar ladder complete $(date) ==="
} >> "$LOG" 2>&1

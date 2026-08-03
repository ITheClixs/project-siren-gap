#!/bin/bash
# Everything that has to wait for the S8 corpus sweep to release the accelerator.
#
#   1. W12  : the phasor-graded reader on raw parameters (docs/prereg/S9.md), MNIST P-random
#   2. S8   : decode the ladder at every completed step budget and score it
#
# Launch detached:  nohup caffeinate -i bash scripts/49_w12_and_s8_decode.sh >/dev/null 2>&1 &
# (macOS has no setsid, and harness-tracked background shells do not survive.)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s8/run_w12_decode.log
mkdir -p results/s8

busy() {
  pgrep -fl "(03_generate_inrbench|11_ladder|37_orbit_intervention|47_w12_phasor|48_s8_sweep)[.]py" \
    2>/dev/null | grep -q "[.]venv/bin/python\|Python.app"
}

{
  echo "=== waiting for the S8 corpora $(date) ==="
  while busy; do sleep 60; done

  # W12 may have been run already, alongside the corpora rather than behind them: the
  # fitting sweep turned out to be ~11.5 h and W12's outcome is the one that can force a
  # headline change, so it is worth having early. Never overwrite a completed cell.
  if [ -f results/ladder/mnist/W12.json ]; then
    echo "--- W12 already present, skipping $(date) ---"
  else
    echo "--- W12: phasor-graded reader, MNIST P-random $(date) ---"
    $PY scripts/47_w12_phasor.py --dataset mnist || echo "W12 FAILED"
  fi

  echo "--- S8: decode the sweep $(date) ---"
  $PY scripts/48_s8_sweep.py --dataset mnist --budgets 300 1000 3000 10000 || echo "S8 DECODE FAILED"

  echo "=== W12 + S8 decode complete $(date) ==="
} >> "$LOG" 2>&1

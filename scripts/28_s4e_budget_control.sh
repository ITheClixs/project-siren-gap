#!/bin/bash
# S4e budget control: is the w=16/32 basin collapse a basin fact or a training-budget artifact?
#
# In the confirmatory run the warm-start fits converged at w<=8 (R_f -> ~1e-6) but not at
# w=32 (R_f 1.5e-3 -> 1.7e-3 in 8000 steps). "0% returned to the orbit" therefore conflates
# a small basin with an under-trained fit. This re-runs the two largest widths at the full
# 40000-step student budget; if recovery stays at 0 the basin reading holds, and if it rises
# the basin claim must be weakened to a statement about optimisation budget.
#
# Exploratory: not part of the frozen registration. Waits for the confirmatory run to exit
# so the two never contend for the MPS device.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s4e/budget_control.log

running() { pgrep -fl "26_s4e_identifiability" 2>/dev/null | grep -q "[.]venv/bin/python\|Python.app"; }

{
  echo "=== budget control: waiting for the confirmatory run $(date) ==="
  while running; do sleep 60; done
  echo "=== confirmatory run finished; starting $(date) ==="
  $PY scripts/26_s4e_identifiability.py \
    --arms warmstart --widths 16 32 --warm-n 32 --warm-steps 40000 \
    --warm-eps 1e-5 1e-4 --lr 2e-3 --grid-side 64 \
    --out results/s4e/budget_control.json || echo "BUDGET CONTROL FAILED"
  echo "=== budget control complete $(date) ==="
} >> "$LOG" 2>&1

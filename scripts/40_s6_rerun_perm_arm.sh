#!/bin/bash
# S6 arm (i), re-run under an explicit tag.
#
# Why: the first launch of arm (i) wrote results/s6/orbit_mnist.json, and a later launch of
# arm (ii) was started *without* --tag, so it overwrote that file with the identity-permutation
# numbers. Only run_perm.log survived. The log's means are the check this re-run must reproduce:
#
#   B=0 raw 15.29 / c_sort 60.59 / c_align 83.67 / invariants 72.51 / delta_sym +79.07
#   B=1 raw 15.31 / B=3 raw 15.57 / B=10 raw 15.27
#
# Same seed (1234) and same seed count, so the reproduction should be exact.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s6/run_perm_rerun.log

busy() {
  pgrep -fl "(37_orbit_intervention|11_ladder)[.]py" 2>/dev/null \
    | grep -q "[.]venv/bin/python\|Python.app"
}

{
  echo "=== waiting for the device $(date) ==="
  while busy; do sleep 60; done

  echo "--- S6 arm (i): permuted, B in {0,1,3,10} $(date) ---"
  $PY scripts/37_orbit_intervention.py --dataset mnist --windings 0 1 3 10 --seeds 5 \
      --tag perm || echo "ARM I RERUN FAILED"

  echo "=== arm (i) re-run complete $(date) ==="
} >> "$LOG" 2>&1

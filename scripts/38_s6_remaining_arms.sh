#!/bin/bash
# S6 arms (iii) and (iv), chained behind the permuted and identity-permutation arms so that
# nothing contends for the MPS device. Arm (ii) is launched by the earlier chain.
#
#   (iii) equivariant readers W11a/W11b on the scattered corpus, at B=3
#   (iv)  the same intervention applied to P-random (is it already group-saturated?)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s6/run_remaining.log

# NOTE: a bare `pgrep -f <literal>` also matches any polling shell whose command line
# contains that literal -- including this one, and including an interactive waiter.
# That deadlocked this chain once. Match the python process itself.
busy() { pgrep -fl "37_orbit_intervention" 2>/dev/null | grep -q "[.]venv/bin/python\|Python.app"; }

{
  echo "=== waiting for the earlier S6 arms $(date) ==="
  while busy; do sleep 60; done
  # the identity-permutation arm is chained separately; wait for it too
  sleep 90
  while busy; do sleep 60; done

  echo "--- arm (iii): equivariant readers at B=3 $(date) ---"
  $PY scripts/37_orbit_intervention.py --dataset mnist --windings 3 --seeds 5 \
      --equivariant --tag equivariant || echo "ARM III FAILED"

  echo "--- arm (iv): the intervention on P-random $(date) ---"
  $PY scripts/37_orbit_intervention.py --dataset mnist --protocol P-random \
      --windings 0 3 --seeds 5 --tag prandom || echo "ARM IV FAILED"

  echo "=== S6 remaining arms complete $(date) ==="
} >> "$LOG" 2>&1

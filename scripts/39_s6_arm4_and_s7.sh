#!/bin/bash
# S6 arm (iv) and the S7 control rung, chained behind whatever is currently on the MPS device.
#
#   (iv)  the orbit intervention applied to P-random (is it already group-saturated?)
#   S7    rung W10c on MNIST and CIFAR-10 (docs/prereg/S7.md)
#
# NOTE (inherited from scripts/38): a bare `pgrep -f <literal>` also matches the polling
# shell whose command line contains that literal, including this one. Match the python
# process itself. NOTE 2: pgrep takes an *extended* regex, so alternation is `|`, not
# `\|` -- the BRE spelling silently matches nothing and the wait falls straight through.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s6/run_arm4_s7.log

busy() {
  pgrep -fl "(37_orbit_intervention|11_ladder)[.]py" 2>/dev/null \
    | grep -q "[.]venv/bin/python\|Python.app"
}

{
  echo "=== waiting for the device $(date) ==="
  while busy; do sleep 60; done

  echo "--- S6 arm (iv): the intervention on P-random $(date) ---"
  $PY scripts/37_orbit_intervention.py --dataset mnist --protocol P-random \
      --windings 0 3 --seeds 5 --tag prandom || echo "ARM IV FAILED"

  echo "--- S7: rung W10c, MNIST $(date) ---"
  $PY scripts/11_ladder.py --dataset mnist --rungs W10c || echo "S7 MNIST FAILED"

  echo "--- S7: rung W10c, CIFAR-10 $(date) ---"
  $PY scripts/11_ladder.py --dataset cifar10 --rungs W10c || echo "S7 CIFAR FAILED"

  echo "=== arm (iv) + S7 complete $(date) ==="
} >> "$LOG" 2>&1

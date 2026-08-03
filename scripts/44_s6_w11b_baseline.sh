#!/bin/bash
# The unscattered W11b baseline that H-S6-5 needs.
#
# H-S6-5 registers |W11b accuracy on the scattered corpus - on the unscattered one| as a validity
# check: W11b's pipeline is G-invariant, so the difference must be seed noise. Arm (iii) supplies
# the scattered side; this supplies the other, on `P-shared-det` with no group applied. It writes
# W11_shareddet.json, never W11.json.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s6/run_w11b_baseline.log

busy() {
  pgrep -fl "(37_orbit_intervention|11_ladder|33_w11_equivariant)[.]py" 2>/dev/null \
    | grep -q "[.]venv/bin/python\|Python.app"
}

{
  echo "=== waiting for the device $(date) ==="
  while busy; do sleep 60; done

  echo "--- W11b on unscattered P-shared-det $(date) ---"
  $PY scripts/33_w11_equivariant.py --dataset mnist --variants b \
      --protocol P-shared-det --out-name W11_shareddet || echo "W11B BASELINE FAILED"

  echo "=== W11b baseline complete $(date) ==="
} >> "$LOG" 2>&1

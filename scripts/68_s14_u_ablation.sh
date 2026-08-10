#!/bin/bash
# S14: the two ablation arms, queued behind whatever is fitting. Arm A is the existing
# results/ladder/cifar10/W12.json and is not re-run.
#
# Launch detached:  nohup caffeinate -i bash scripts/68_s14_u_ablation.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/ladder/cifar10/run_s14.log

while pgrep -f "(03_generate_inrbench|47_w12_phasor)[.]py" >/dev/null 2>&1; do sleep 120; done

step() { local l="$1"; shift; echo "--- $l $(date) ---"; "$@" || echo "STEP FAILED: $l"; }
{
  echo "=== S14 start $(date) ==="
  step "arm B: collapse u to its channel mean" \
    $PY scripts/47_w12_phasor.py --dataset cifar10 --u-mode mean --out-name W12_umean
  step "arm C: pad that mean back to c channels" \
    $PY scripts/47_w12_phasor.py --dataset cifar10 --u-mode mean_pad --out-name W12_umeanpad
  echo "=== S14 complete $(date) ==="
} >> "$LOG" 2>&1

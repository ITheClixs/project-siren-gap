#!/bin/bash
# S11: the fourth cell of the 2x2, then W12 on the three corpora it was not designed on.
# Registered in docs/prereg/S11.md before launch. Serial, because the accelerator is one GPU and
# contention halved throughput last time.
#
# Launch detached:  nohup caffeinate -i bash scripts/57_s11_chain.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/ladder/run_s11.log

step() { local label="$1"; shift; echo "--- $label $(date) ---"; "$@" || echo "STEP FAILED: $label"; }

{
  echo "=== S11 chain start $(date) ==="
  while pgrep -f "(47_w12_phasor|03_generate_inrbench|11_ladder)[.]py" >/dev/null 2>&1; do sleep 60; done

  if [ -f results/ladder/mnist/W12ub.json ]; then
    echo "--- W12ub present, skipping ---"
  else
    step "W12ub: the fourth cell (ungraded skeleton, raw bias)" \
      $PY scripts/47_w12_phasor.py --dataset mnist --ungraded --raw-bias
  fi

  for ds in fashionmnist cifar10gray cifar10; do
    if [ -f "results/ladder/$ds/W12.json" ]; then
      echo "--- W12 on $ds present, skipping ---"
      continue
    fi
    step "W12 on $ds" $PY scripts/47_w12_phasor.py --dataset "$ds"
    step "invariance audit on $ds" \
      $PY scripts/52_w12_invariance_audit.py --dataset "$ds"
  done

  echo "=== S11 chain complete $(date) ==="
} >> "$LOG" 2>&1

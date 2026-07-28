#!/bin/bash
# CIFAR-10 sine corpus at the frozen pilot config (w32 L2, steps 1000), 4 protocols + gates.
# Config frozen by the rule in docs/THINKING/G3-cifar-pilot.md: cheapest sweep config whose
# task-referenced gate passes. Path (full vs 20k/4k fallback) decided by rule R-CIFAR; pass
# FALLBACK=1 to take the reduced path.
#
# Detached usage:
#   nohup caffeinate -i bash scripts/16_cifar_corpus.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/cifar_corpus.log
FALLBACK="${FALLBACK:-0}"
mkdir -p results

# NOTE: plain `pgrep -f <pattern>` also matches *any shell whose command line contains the
# pattern* - including a waiter loop that greps for it. That deadlocked the R-CIFAR decision job
# on 2026-07-28 (it waited on a phantom that was itself). Match the python process instead, and
# never embed these literals in a polling shell one-liner.
gen_running() { pgrep -fl "03_generate_inrbench" 2>/dev/null | grep -q "[.]venv/bin/python"; }

GEN="scripts/03_generate_inrbench.py --dataset cifar10 --steps 1000 --width 32 --layers 2 --batch 256"
if [ "$FALLBACK" = "1" ]; then
  SUBSET="--n-train 20000 --n-val 2000 --n-test 2000"
else
  SUBSET=""
fi

{
  echo "=== cifar corpus start $(date) (fallback=$FALLBACK) ==="
  while gen_running; do sleep 60; done
  for p in P-shared-det P-random P-shared-stoch; do
    $PY $GEN --protocol "$p" --split all $SUBSET && echo "CIFAR $p DONE $(date)"
  done
  $PY $GEN --protocol P-random-K --split train $SUBSET && echo "CIFAR K DONE $(date)"

  echo "=== gates $(date) ==="
  for p in P-shared-det P-random P-shared-stoch; do
    $PY scripts/04_quality_gate.py --dir "data/inrbench/cifar10/$p" --dataset cifar10 \
        --eval-split test --gate-epochs 10 || echo "GATE FAILED $p"
  done
  echo "=== cifar corpus complete $(date) ==="
} >> "$LOG" 2>&1

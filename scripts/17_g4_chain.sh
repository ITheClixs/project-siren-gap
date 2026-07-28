#!/bin/bash
# G4 chain, in cost order: W5 template sensitivity (exploratory) -> S1 ladder on FashionMNIST
# (replication of the MNIST ladder, identical analysis) -> full CIFAR-10 corpus generation.
# Serialized: everything here contends for the same MPS device.
#
# Detached usage:
#   nohup caffeinate -i bash scripts/17_g4_chain.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/g4_chain.log
mkdir -p results

# NOTE: plain `pgrep -f <pattern>` also matches *any shell whose command line contains the
# pattern* - including a waiter loop that greps for it. That deadlocked the R-CIFAR decision job
# on 2026-07-28 (it waited on a phantom that was itself). Match the python process instead, and
# never embed these literals in a polling shell one-liner.
gen_running() { pgrep -fl "03_generate_inrbench" 2>/dev/null | grep -q "[.]venv/bin/python"; }

RUNGS="P0 P1 W1 W2 W3 W4 W5 W10 W9 X1 W6 W8 W7-1/8 W7"

{
  echo "=== g4 chain start $(date) ==="
  while gen_running; do sleep 60; done

  echo "--- W5 template sensitivity (exploratory) $(date) ---"
  $PY scripts/15_w5_template_sensitivity.py --dataset mnist || echo "SENSITIVITY FAILED"

  echo "--- S1 ladder on fashionmnist $(date) ---"
  for r in $RUNGS; do
    echo "--- rung $r $(date) ---"
    $PY scripts/11_ladder.py --dataset fashionmnist --rungs "$r" || echo "RUNG FAILED $r"
  done
  $PY scripts/14_ladder_analysis.py --dataset fashionmnist || echo "ANALYSIS FAILED"
  echo "--- fashionmnist ladder done $(date) ---"

  echo "--- CIFAR-10 corpus $(date) ---"
  bash scripts/16_cifar_corpus.sh
  echo "=== g4 chain complete $(date) ==="
} >> "$LOG" 2>&1

#!/bin/bash
# Grayscale CIFAR-10 corpora at the RGB-CIFAR frozen config (w32 L2, 1000 steps, lr 1e-3).
#
# Purpose (prereg docs/prereg/S1-gray.md): CIFAR-10 differs from the grayscale corpora in image
# statistics AND in output-channel count, and every cross-dataset claim in the paper is confounded
# between the two. Luminance CIFAR holds the images, the architecture and the fit budget fixed and
# changes only c: 3 -> 1.
#
# Two protocols only. P-shared-det carries P0/P1/W1; P-random carries W3/W4/W5/W9/W10 — the rungs
# that decide the question. P-shared-stoch (W2) and P-random-K (W7) are not run; the ladder is
# reported as partial, which prereg S1 section 6 permits.
#
# Detached usage:
#   nohup caffeinate -i bash scripts/30_cifar_gray_corpus.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/cifar_gray_corpus.log
GEN="scripts/03_generate_inrbench.py --dataset cifar10gray --steps 1000 --width 32 --layers 2 --batch 256"
mkdir -p results

{
  echo "=== grayscale CIFAR corpus start $(date) ==="
  for proto in P-shared-det P-random; do
    echo "--- $proto $(date) ---"
    $PY $GEN --protocol "$proto" --split all || { echo "GEN FAILED $proto"; exit 1; }
  done
  echo "--- quality gates $(date) ---"
  for proto in P-shared-det P-random; do
    $PY scripts/04_quality_gate.py --dataset cifar10gray --corpus "data/inrbench/cifar10gray/$proto" \
      --split test --epochs 10 || echo "GATE FAILED $proto"
  done
  echo "=== grayscale CIFAR corpus complete $(date) ==="
} >> "$LOG" 2>&1

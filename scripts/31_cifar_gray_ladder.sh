#!/bin/bash
# Partial S1 ladder on luminance CIFAR-10, against the frozen registration
# docs/prereg/S1-gray.md (sha256-16 b84b660829aa6d40).
#
# Two protocols exist, so eight rungs run: P0/P1/W1 from P-shared-det and W3/W4/W5/W9/W10 from
# P-random. W2, W6, W7, W8 are not run and the ladder is reported as partial (S1 section 6).
#
# Waits for generation so the two never contend for the MPS device.
#
# Detached usage:
#   nohup caffeinate -i bash scripts/31_cifar_gray_ladder.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/cifar_gray_ladder.log
RUNGS="P0 P1 W1 W3 W4 W5 W10 W9"
mkdir -p results

gen_running() { pgrep -fl "03_generate_inrbench|04_quality_gate" 2>/dev/null | grep -q "[.]venv/bin/python\|Python.app"; }

{
  echo "=== waiting for the grayscale corpus $(date) ==="
  while gen_running; do sleep 60; done
  if ! grep -q "grayscale CIFAR corpus complete" results/cifar_gray_corpus.log 2>/dev/null; then
    echo "!! corpus did not report complete; not starting the ladder"; exit 1
  fi
  echo "=== corpus complete; ladder start $(date) ==="
  for r in $RUNGS; do
    echo "--- rung $r $(date) ---"
    $PY scripts/11_ladder.py --dataset cifar10gray --rungs "$r" || echo "RUNG FAILED $r"
  done
  echo "--- analysis $(date) ---"
  $PY scripts/14_ladder_analysis.py --dataset cifar10gray || echo "ANALYSIS FAILED"
  echo "=== cifar10gray ladder complete $(date) ==="
} >> "$LOG" 2>&1

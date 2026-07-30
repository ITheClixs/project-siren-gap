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
  # Gate on the *gate*, not on the corpus log: a corpus that has not passed its quality gate is
  # not admissible (S1 section 6), and the first run of this script started the ladder on
  # ungated corpora because it only checked that generation had finished (deviation D1).
  for proto in P-shared-det P-random; do
    g="results/inrbench/cifar10gray_${proto}_test_gate.json"
    if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$g'))['gate']['passes'] else 1)" 2>/dev/null; then
      echo "!! $proto has no passing quality gate ($g); not starting the ladder"; exit 1
    fi
  done
  echo "=== corpus complete; ladder start $(date) ==="
  for r in $RUNGS; do
    echo "--- rung $r $(date) ---"
    $PY scripts/11_ladder.py --dataset cifar10gray --rungs "$r" || echo "RUNG FAILED $r"
  done
  echo "--- analysis $(date) ---"
  $PY scripts/14_ladder_analysis.py --dataset cifar10gray || echo "ANALYSIS FAILED"
  echo "=== cifar10gray ladder complete $(date) ==="
} >> "$LOG" 2>&1

#!/bin/bash
# S8 convergence sweep: corpora at four step budgets, two protocols (docs/prereg/S8.md).
#
# The review's Priority 6. Every ladder result in this paper is measured on fits that end
# unconverged -- the microcosm ends with a relative gradient norm around 5e-3 at the corpus
# budget -- so the recoverable fraction may be a property of the early-stopped regime rather
# than of independently fitted INRs generally. This sweep fits the same images at 300, 1000,
# 3000 and 10000 steps and decodes the ladder at each.
#
# Corpus size is reduced to 10k/2k/2k so that the 10000-step arm is affordable; the 300-step
# arm at the *same* size is the internal control, not the full-corpus ladder.
#
# Measured cost: ~6e-5 s per fit-step, so 14k INRs cost 4 / 14 / 41 / 138 min per protocol,
# ~3.4 h per protocol and ~6.7 h in total.
#
# Launch detached:  nohup caffeinate -i bash scripts/46_s8_corpora.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s8/run_corpora.log
mkdir -p results/s8

busy() {
  pgrep -fl "(37_orbit_intervention|11_ladder|33_w11_equivariant|03_generate_inrbench)[.]py" \
    2>/dev/null | grep -q "[.]venv/bin/python\|Python.app"
}

{
  echo "=== S8 corpora start $(date) ==="
  while busy; do sleep 60; done

  for steps in 300 1000 3000 10000; do
    for protocol in P-shared-det P-random; do
      echo "--- $protocol @ $steps steps $(date) ---"
      $PY scripts/03_generate_inrbench.py --dataset mnist --protocol "$protocol" \
          --steps "$steps" --n-train 10000 --n-val 2000 --n-test 2000 \
          --tag "s8s$steps" || echo "FAILED: $protocol @ $steps"
    done
  done

  echo "=== S8 corpora complete $(date) ==="
} >> "$LOG" 2>&1

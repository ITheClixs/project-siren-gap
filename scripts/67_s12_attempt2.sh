#!/bin/bash
# S12: the converged-fit ladder. Three independent corpus generations per protocol, fitted with a
# cosine-decayed schedule and a per-INR stationarity stop, then the four decoded arms.
#
# Registered in docs/prereg/S12.md before the fitter was written. The validity conditions of its
# section 3 are checked by 60_score_s12.py BEFORE any ladder number is quoted; if they fail, the
# study reports that failure and no number from these corpora enters the paper.
#
# Launch detached:  nohup caffeinate -i bash scripts/59_s12_converged.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s12/run_attempt2.log
mkdir -p results/s12

STEPS=12000         # attempt 2 (S12-addendum-01): the bar is unchanged, the cap doubles
TOL=1e-4            # the stationarity tolerance of S12 section 2
LRF=1e-6

step() { local label="$1"; shift; echo "--- $label $(date) ---"; "$@" || echo "STEP FAILED: $label"; }

{
  echo "=== S12 chain start $(date) ==="
  while pgrep -f "(03_generate_inrbench|47_w12_phasor|11_ladder)[.]py" >/dev/null 2>&1; do sleep 60; done

  for rep in 0; do   # staged: replication 0 alone decides whether 1 and 2 are fitted
    for protocol in P-shared-det P-random; do
      tag="s12b${rep}"
      if [ -d "data/inrbench/mnist/${protocol}-${tag}" ] && \
         [ "$(ls "data/inrbench/mnist/${protocol}-${tag}"/*.parquet 2>/dev/null | wc -l)" -ge 55 ]; then
        echo "--- ${protocol} ${tag} already complete, skipping ---"
        continue
      fi
      step "${protocol} replication ${rep}, converged" \
        $PY scripts/03_generate_inrbench.py --dataset mnist --protocol "$protocol" \
            --steps "$STEPS" --schedule cosine --lr-final "$LRF" --stop-grad-norm "$TOL" \
            --n-train 10000 --n-val 2000 --n-test 2000 --tag "$tag" \
            --seed-offset $((7000 * (rep + 1)))
    done
  done

  echo "=== S12 corpora complete $(date); decoding is a separate, gated step ==="
} >> "$LOG" 2>&1

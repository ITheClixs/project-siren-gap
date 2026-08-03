#!/bin/bash
# One serial chain for everything S8 and S9 still owe, in the order that gets the
# consequential result first.
#
#   1. (already running outside this chain) W12, the phasor-graded reader
#   2. W12u, its matched ungraded control
#   3. the S8 corpora, resumed -- 03_generate_inrbench skips completed shards, so pausing it
#      to let W12 run alone cost nothing but the shard that was in flight
#   4. the S8 decode and scoring
#
# Why serialize: running W12 alongside the fitter dropped the fitter from 17 to 4.3 fits/s and
# W12 did not finish a seed in 20 minutes. Contention roughly halved total throughput, and the
# corpora are resumable, so the accelerator is given to one job at a time.
#
# Launch detached:  nohup caffeinate -i bash scripts/51_master_chain_s8_s9.sh >/dev/null 2>&1 &
# (macOS has no setsid; harness-tracked background shells do not survive.)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s8/run_master.log
mkdir -p results/s8

busy() {
  pgrep -fl "(03_generate_inrbench|11_ladder|37_orbit_intervention|47_w12_phasor|48_s8_sweep)[.]py" \
    2>/dev/null | grep -q "[.]venv/bin/python\|Python.app"
}

step() { local label="$1"; shift; echo "--- $label $(date) ---"; "$@" || echo "STEP FAILED: $label"; }

{
  echo "=== master chain start $(date) ==="
  while busy; do sleep 60; done

  if [ -f results/ladder/mnist/W12u.json ]; then
    echo "--- W12u already present, skipping $(date) ---"
  else
    step "W12u: the matched ungraded control" \
      $PY scripts/47_w12_phasor.py --dataset mnist --ungraded
  fi

  # The S6 triple that licensed "the gap is not reducible to symmetry" was measured with W11b.
  # W12 recovers far more of the same gap while being equally invariant, so the shared-init leg
  # of that triple has to be re-measured with the better reader before the claim can stand.
  if [ -f results/ladder/mnist/W12_shareddet.json ]; then
    echo "--- W12 on P-shared-det already present, skipping $(date) ---"
  else
    step "W12 on unscattered P-shared-det (re-measures the S6 triple)" \
      $PY scripts/47_w12_phasor.py --dataset mnist --protocol P-shared-det \
          --out-name W12_shareddet
  fi

  # Budgets in increasing order, and the decode re-run after 3000 as well as at the end. The
  # 10000-step arm is ~8 h of the ~11 h total, while 1000 steps already drops the relative
  # gradient norm 24x (5.5e-3 -> 2.25e-4) at 69 dB, so the convergence question is largely
  # answered by 3000. Decoding early gives that answer without deviating from the registration,
  # which still commits to all four budgets. 48_s8_sweep is idempotent and skips missing arms.
  DONE=""
  for steps in 300 1000 3000 10000; do
    for protocol in P-shared-det P-random; do
      step "$protocol @ $steps steps" \
        $PY scripts/03_generate_inrbench.py --dataset mnist --protocol "$protocol" \
            --steps "$steps" --n-train 10000 --n-val 2000 --n-test 2000 --tag "s8s$steps"
    done
    DONE="$DONE $steps"
    if [ "$steps" = "3000" ] || [ "$steps" = "10000" ]; then
      step "S8: decode budgets$DONE" \
        $PY scripts/48_s8_sweep.py --dataset mnist --budgets $DONE
    fi
  done

  echo "=== master chain complete $(date) ==="
} >> "$LOG" 2>&1

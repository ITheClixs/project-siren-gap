#!/bin/bash
# Resume of scripts/51_master_chain_s8_s9.sh, which died mid-way through the 10000-step
# P-shared-det corpus (last shard written: shard_011008, 08:43). Everything before that step
# completed and is on disk: W12u, W12 on P-shared-det, the S8 decode at 300/1000/3000.
#
# What is left:
#   1. P-shared-det @ 10000 steps -- resumes, 03_generate_inrbench skips completed shards
#   2. P-random     @ 10000 steps
#   3. the S8 decode over all four budgets, which is what the registration commits to
#   4. S5 re-priced at the REGISTERED seed count
#
# Why (4) is here: the chain's S5 step ran 35_s5_pareto.py at its default --seeds 3, but S5 §6
# registers 5 seeds for the headline K sweep (the stopping rule permits 3 only for K=256, and
# only if the sweep exceeds 3 h, which it did not). That off-protocol run overwrote the compliant
# artifact and its re-score appended a second copy of all 11 S5 rows to the ledger. The artifact
# has been reverted and the duplicate rows removed; this step regenerates it at n=5 with W12 on
# the frontier, which is the only thing the re-price was for.
#
# Launch detached:  nohup caffeinate -i bash scripts/53_resume_s8_decodes.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s8/run_master.log

busy() {
  pgrep -fl "(03_generate_inrbench|11_ladder|37_orbit_intervention|47_w12_phasor|48_s8_sweep|35_s5_pareto)[.]py" \
    2>/dev/null | grep -q "[.]venv/bin/python\|Python.app"
}

step() { local label="$1"; shift; echo "--- $label $(date) ---"; "$@" || echo "STEP FAILED: $label"; }

{
  echo "=== resume chain start $(date) ==="
  while busy; do sleep 60; done

  for protocol in P-shared-det P-random; do
    step "$protocol @ 10000 steps" \
      $PY scripts/03_generate_inrbench.py --dataset mnist --protocol "$protocol" \
          --steps 10000 --n-train 10000 --n-val 2000 --n-test 2000 --tag "s8s10000"
  done

  step "S8: decode all four budgets" \
    $PY scripts/48_s8_sweep.py --dataset mnist --budgets 300 1000 3000 10000

  step "S5: re-price the frontier with W12, at the registered n=5" \
    $PY scripts/35_s5_pareto.py --dataset mnist --seeds 5 --nuisance-control --frozen-ablation

  echo "=== resume chain complete $(date) ==="
} >> "$LOG" 2>&1

#!/bin/bash
# One sequential chain for everything still owed on S6 and S7, so nothing contends for the
# accelerator and nothing races another waiter.
#
#   1. S6 arm (iv)  : the intervention on P-random (is it already group-saturated?)
#   2. S7           : rung W10c on MNIST
#   3. S7           : rung W10c on CIFAR-10
#   4. S6 arm (i)   : the permuted arm, re-run under an explicit tag (its JSON was overwritten)
#   5. S6 H-S6-5    : W11b on the unscattered P-shared-det corpus
#
# Launch detached:  nohup caffeinate -i bash scripts/45_s6_s7_master_chain.sh >/dev/null 2>&1 &
#
# NOTES, both learned the hard way:
#   * there is no `setsid` on macOS, so `nohup setsid bash ...` silently runs nothing.
#   * `pgrep` takes an *extended* regex; the BRE spelling `\|` matches nothing and the wait
#     loop falls straight through, starting a second job on a busy device.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s6/run_master_chain.log

busy() {
  pgrep -fl "(37_orbit_intervention|11_ladder|33_w11_equivariant)[.]py" 2>/dev/null \
    | grep -q "[.]venv/bin/python\|Python.app"
}

step() {  # step <label> <command...>
  local label="$1"; shift
  echo "--- $label $(date) ---"
  "$@" || echo "STEP FAILED: $label"
}

{
  echo "=== master chain start $(date) ==="
  while busy; do sleep 60; done

  step "S6 arm (iv): the intervention on P-random" \
    $PY scripts/37_orbit_intervention.py --dataset mnist --protocol P-random \
        --windings 0 3 --seeds 5 --tag prandom

  step "S7: rung W10c, MNIST" \
    $PY scripts/11_ladder.py --dataset mnist --rungs W10c

  step "S7: rung W10c, CIFAR-10" \
    $PY scripts/11_ladder.py --dataset cifar10 --rungs W10c

  step "S6 arm (i): permuted, re-run under --tag perm" \
    $PY scripts/37_orbit_intervention.py --dataset mnist --windings 0 1 3 10 --seeds 5 --tag perm

  step "S6 H-S6-5: W11b on unscattered P-shared-det" \
    $PY scripts/33_w11_equivariant.py --dataset mnist --variants b \
        --protocol P-shared-det --out-name W11_shareddet

  echo "=== master chain complete $(date) ==="
} >> "$LOG" 2>&1

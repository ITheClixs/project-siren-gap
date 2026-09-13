#!/bin/bash
# S15 (factorial orbit intervention), S16 (function-side control) and S17 (theta_0 replicates), one
# after another behind any running W12 job, because two jobs on one GPU halve throughput.
#
# Launch detached:  nohup caffeinate -i bash scripts/78_s15_s17_chain.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

while pgrep -f "(47_w12_phasor|74_bench_w0_chain)[.](py|sh)" >/dev/null 2>&1; do sleep 60; done

for study in s15 s16 s17; do mkdir -p results/$study; done
{ echo "=== S15 $(date) ==="; $PY scripts/75_s15_orbit_factorial.py --dataset mnist; echo "=== done $(date) ==="; } \
  >> results/s15/run.log 2>&1
{ echo "=== S16 $(date) ==="; $PY scripts/76_s16_function_nuisance.py --dataset mnist; echo "=== done $(date) ==="; } \
  >> results/s16/run.log 2>&1
{ echo "=== S17 $(date) ==="; $PY scripts/77_s17_theta0.py --dataset mnist; echo "=== done $(date) ==="; } \
  >> results/s17/run.log 2>&1

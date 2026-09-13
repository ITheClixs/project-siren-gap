#!/bin/bash
# W12 on the three published INR benchmarks after omega_0 absorption (script 73), five seeds each,
# one dataset at a time because two fitters on one GPU halve throughput. Writes W12_dwsbench_w0.json
# beside the original W12_dwsbench.json; the original cells are left untouched for comparison.
#
# Launch detached:  nohup caffeinate -i bash scripts/74_bench_w0_chain.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

while pgrep -f "47_w12_phasor[.]py" >/dev/null 2>&1; do sleep 60; done

for ds in mnist fashionmnist cifar10; do
  LOG=results/ladder/$ds/run_w12_dwsbench_w0.log
  mkdir -p results/ladder/$ds
  {
    echo "=== W12 on the $ds INR benchmark, omega_0 absorbed, $(date) ==="
    $PY scripts/47_w12_phasor.py --dataset $ds --protocol P-dws-bench-w0 \
        --out-name W12_dwsbench_w0
    echo "=== done $(date) ==="
  } >> "$LOG" 2>&1
done

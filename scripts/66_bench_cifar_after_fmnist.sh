#!/bin/bash
# Queues W12 on the CIFAR-10 INR benchmark behind the FashionMNIST run, because running two
# fitters on one GPU halved throughput the last time this program tried it.
#
# Launch detached:  nohup caffeinate -i bash scripts/66_bench_cifar_after_fmnist.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/ladder/cifar10/run_w12_dwsbench.log
mkdir -p results/ladder/cifar10

while pgrep -f "47_w12_phasor[.]py" >/dev/null 2>&1; do sleep 60; done

{
  echo "=== W12 on the CIFAR-10 INR benchmark $(date) ==="
  $PY scripts/47_w12_phasor.py --dataset cifar10 --protocol P-dws-bench \
      --out-name W12_dwsbench
  echo "=== done $(date) ==="
} >> "$LOG" 2>&1

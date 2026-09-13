#!/bin/bash
# S15 addendum 01 (the factorial on FashionMNIST and CIFAR-10), then S18 (W12 at width 128), one
# job at a time.
#
# Launch detached:  nohup caffeinate -i bash scripts/79_s15x_s18_chain.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

while pgrep -f "(47_w12_phasor|75_s15_orbit_factorial|76_s16|77_s17)[.]py" >/dev/null 2>&1; do sleep 60; done

mkdir -p results/s15 results/ladder_w128/mnist
for ds in fashionmnist cifar10; do
  { echo "=== S15 $ds $(date) ==="; $PY scripts/75_s15_orbit_factorial.py --dataset $ds; echo "=== done $(date) ==="; } \
    >> results/s15/run_$ds.log 2>&1
done

LOG=results/ladder_w128/mnist/run_s18.log
{
  echo "=== S18 W12 on P-random-w128 $(date) ==="
  $PY scripts/47_w12_phasor.py --dataset mnist --protocol P-random-w128 \
      --anchors-dir results/ladder_w128/mnist --out-dir results/ladder_w128/mnist --out-name W12_w128
  echo "=== S18 W12 on P-shared-det-w128 $(date) ==="
  $PY scripts/47_w12_phasor.py --dataset mnist --protocol P-shared-det-w128 \
      --anchors-dir results/ladder_w128/mnist --out-dir results/ladder_w128/mnist --out-name W12_w128_shareddet
  echo "=== done $(date) ==="
} >> "$LOG" 2>&1

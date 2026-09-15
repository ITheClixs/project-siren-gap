#!/usr/bin/env bash
# S19 GPU chain (docs/prereg/S19.md): A1, A3, A6, A4, A2, one after another on the single device.
# A failed arm is logged and the chain moves on; nothing here overwrites a registered cell.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s19/run_chain.log
mkdir -p results/s19
export PYTHONDONTWRITEBYTECODE=1

run() {
    local label="$1"; shift
    echo "=== $label $(date)" >> "$LOG"
    "$@" >> "$LOG" 2>&1 || echo "!!! FAILED: $label" >> "$LOG"
}

echo "=== S19 chain start $(date)" >> "$LOG"
run "A1 W12 width 32 10k P-random" $PY scripts/47_w12_phasor.py --dataset mnist \
    --protocol P-random-s8s300 --anchors-dir results/s19/anchors_10k --out-dir results/s19 \
    --out-name W12_w32_10k_random --prereg docs/prereg/S19.md
run "A1 W12 width 32 10k P-shared-det" $PY scripts/47_w12_phasor.py --dataset mnist \
    --protocol P-shared-det-s8s300 --anchors-dir results/s19/anchors_10k --out-dir results/s19 \
    --out-name W12_w32_10k_shareddet --prereg docs/prereg/S19.md
if $PY scripts/80_s19_dose_response.py --dataset mnist --limit 300 --seeds 1 >> "$LOG" 2>&1; then
    run "A3 dose response" $PY scripts/80_s19_dose_response.py --dataset mnist
else
    echo "!!! A3 smoke test failed; dose arm skipped" >> "$LOG"
fi
run "A6 fold then grade" $PY scripts/47_w12_phasor.py --dataset mnist --protocol P-random \
    --fold-bias --out-dir results/s19 --out-name W12fb_mnist --prereg docs/prereg/S19.md
run "A4 grid-fixed probes" $PY scripts/76_s16_function_nuisance.py --dataset mnist \
    --probes 16 64 --grid-probes
run "A2 W12 FashionMNIST P-shared-det" $PY scripts/47_w12_phasor.py --dataset fashionmnist \
    --protocol P-shared-det --out-dir results/s19 --out-name W12_shareddet_fashionmnist \
    --prereg docs/prereg/S19.md
run "A2 W12 CIFAR-10 P-shared-det" $PY scripts/47_w12_phasor.py --dataset cifar10 \
    --protocol P-shared-det --out-dir results/s19 --out-name W12_shareddet_cifar10 \
    --prereg docs/prereg/S19.md
echo "=== S19 chain done $(date)" >> "$LOG"

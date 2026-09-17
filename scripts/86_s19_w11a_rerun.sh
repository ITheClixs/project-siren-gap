#!/usr/bin/env bash
# S19 addendum 01: W11a with pooled edge statistics, run after the S19 chain and the A9 published
# readers have released the Apple GPU.
set -u
cd "$(dirname "$0")/.."
LOG=results/s19/run_w11a_fixednorm.log
while pgrep -f "83_s19_chain.sh" > /dev/null || pgrep -f "run_published_after_chain.sh" > /dev/null; do
    sleep 60
done
echo "=== S19 addendum 01 W11a fixed normalization $(date)" >> "$LOG"
caffeinate -i .venv/bin/python scripts/33_w11_equivariant.py --dataset mnist --variants a \
    --prereg docs/prereg/S19-addendum-01.md --out-name W11a_fixednorm >> "$LOG" 2>&1
echo "=== done $(date)" >> "$LOG"

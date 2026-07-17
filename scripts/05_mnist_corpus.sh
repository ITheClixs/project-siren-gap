#!/bin/bash
# Full MNIST sine corpus at the frozen config (w32 L2 steps300, G3 pilot).
# Shard-resumable; safe to rerun after interruption. Wrap in caffeinate.
set -u
PY=.venv/bin/python
GEN="scripts/03_generate_inrbench.py --dataset mnist --steps 300 --width 32 --layers 2 --batch 256"

$PY $GEN --protocol P-shared-det   --split all   || exit 1
$PY $GEN --protocol P-random       --split all   || exit 1
$PY $GEN --protocol P-shared-stoch --split all   || exit 1
$PY $GEN --protocol P-random-K     --split train || exit 1
echo "MNIST corpus complete"

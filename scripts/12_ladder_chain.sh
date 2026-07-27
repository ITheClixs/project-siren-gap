#!/bin/bash
# G4 chain: the S1 ladder on MNIST, then the CIFAR-10 pilot sweep once its download lands.
# Serialized on purpose - generation and decoder runs share the MPS device, and the pilot's
# throughput feeds the R-CIFAR decision rule (docs/THINKING/G3-cifar-pilot.md).
#
# Detached usage (survives machine sleep):
#   nohup caffeinate -i bash scripts/12_ladder_chain.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/ladder_chain.log
mkdir -p results

# cheapest and most central rungs first, then the augmentation-bearing ones (15 seeds),
# then W7 (8x training rows, heaviest in memory)
RUNGS="P0 P1 W2 W3 W4 W5 W10 W9 X1 W6 W8 W7-1/8 W7"

{
  echo "=== ladder chain start $(date) ==="
  for r in $RUNGS; do
    echo "--- rung $r $(date) ---"
    $PY scripts/11_ladder.py --dataset mnist --rungs "$r" || echo "RUNG FAILED $r"
  done
  echo "=== ladder complete $(date) ==="

  echo "=== waiting for CIFAR-10 download $(date) ==="
  until $PY -c "
import sys; sys.path.insert(0,'src')
from sirengap.data.images import CIFAR_MEMBERS, DATA_DIR
d = DATA_DIR / 'cifar10' / 'cifar-10-batches-py'
sys.exit(0 if all((d / m).exists() for m in CIFAR_MEMBERS) else 1)
"; do sleep 60; done
  echo "=== CIFAR present, starting pilot $(date) ==="
  bash scripts/09_cifar_pilot.sh
  echo "=== chain complete $(date) ==="
} >> "$LOG" 2>&1

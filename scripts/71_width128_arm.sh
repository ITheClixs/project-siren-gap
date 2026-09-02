#!/bin/bash
# S13: does the pattern survive a materially larger network?
# Same images, same depth, same fitting regime, width 32 -> 128 (1185 -> 17025 parameters).
# Corpus is reduced to 10k/2k/2k so the arm is affordable; the S8 sweep uses the same size, so
# comparisons are within this arm rather than against the full-corpus ladder.
set -eu
cd "$(dirname "$0")/.."
STEPS=${STEPS:-300}
for PROTO in P-shared-det P-random; do
  # A directory is not a finished corpus. Resume unless the expected shard count is present;
  # the generator itself skips shards it has already written.
  DIR="data/inrbench/mnist/${PROTO}-w128"
  WANT=$(( (10000 + 2000 + 2000 + 255) / 256 ))
  HAVE=$(ls "$DIR" 2>/dev/null | grep -c safetensors || true)
  if [ "$HAVE" -ge "$WANT" ]; then
    echo "skip $PROTO, $HAVE/$WANT shards present"; continue
  fi
  [ "$HAVE" -gt 0 ] && echo "resuming $PROTO at $HAVE/$WANT shards"
  echo "=== $PROTO width 128  $(date +%H:%M) ==="
  .venv/bin/python scripts/03_generate_inrbench.py --dataset mnist \
    --protocol "$PROTO" --steps "$STEPS" --width 128 --layers 2 \
    --n-train 10000 --n-val 2000 --n-test 2000 --tag w128
done
echo "=== corpora done $(date +%H:%M) ==="

# --- analysis, once both corpora exist -------------------------------------------------
# The ladder anchors and the exact treatments, then the orbit intervention, then the reader.
if [ -d data/inrbench/mnist/P-random-w128 ]; then
  echo "=== ladder (w128) $(date +%H:%M) ==="
  .venv/bin/python scripts/11_ladder.py --dataset mnist --rungs W1 W3 W4 W5 W10 \
    --shared-protocol P-shared-det-w128 --random-protocol P-random-w128 \
    --out results/ladder_w128
  echo "=== orbit intervention (w128) $(date +%H:%M) ==="
  .venv/bin/python scripts/37_orbit_intervention.py --dataset mnist \
    --protocol P-shared-det-w128 --windings 0 3 --tag w128
  echo "=== phasor-graded reader (w128) $(date +%H:%M) ==="
  .venv/bin/python scripts/47_w12_phasor.py --dataset mnist --protocol P-random-w128 \
    --anchors-dir results/ladder_w128/mnist --out-name W12_w128
  echo "=== w128 arm complete $(date +%H:%M) ==="
fi

#!/bin/bash
# S12 decode. Runs only after 60_score_s12.py --gate records a pass, because S12 section 5
# commits that if the corpora are not converged, nothing is decoded and the failure is what
# gets reported. The gate is checked here as well as inside the scorer, so a decode cannot be
# started by hand against corpora the gate rejected.
#
# Four arms per replication: W1 and W3 fix that replication's own anchors, W5 is the reframing,
# W12 the group-aware reader. Each is decoded from the replication's own corpora, never against
# the anchors of the frozen non-converged ladder.
#
# Launch detached:  nohup caffeinate -i bash scripts/61_s12_decode.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/s12/run_decode.log
OUT=results/s12/ladder
mkdir -p "$OUT"

if ! $PY -c "
import json,sys
from pathlib import Path
g=Path('results/s12/gate.json')
sys.exit(0 if g.exists() and json.loads(g.read_text()).get('gate_passed') else 1)
"; then
  echo "S12 gate has not passed; refusing to decode (prereg section 5)." | tee -a "$LOG"
  exit 1
fi

step() { local label="$1"; shift; echo "--- $label $(date) ---"; "$@" || echo "STEP FAILED: $label"; }

{
  echo "=== S12 decode start $(date) ==="
  for rep in 0 1 2; do
    SH="P-shared-det-s12r${rep}"
    RD="P-random-s12r${rep}"
    step "r${rep} W1/W3/W5 (frozen matched MLP)" \
      $PY scripts/11_ladder.py --dataset mnist --rungs W1 W3 W5 \
          --shared-protocol "$SH" --random-protocol "$RD" \
          --out "$OUT" --flat-out --out-prefix "r${rep}_"
    step "r${rep} W12 (phasor-graded reader)" \
      $PY scripts/47_w12_phasor.py --dataset mnist --protocol "$RD" \
          --anchors-dir "$OUT" --anchors-prefix "r${rep}_" \
          --out-dir "$OUT" --out-name "r${rep}_W12"
  done
  echo "=== S12 decode complete $(date) ==="
} >> "$LOG" 2>&1

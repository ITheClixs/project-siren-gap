#!/bin/bash
# Applies rule R-CIFAR (docs/THINKING/G3-cifar-pilot.md) once the pilot sweep is done:
# takes a clean uncontended throughput measurement at the frozen config, then computes the
# projected hours for the full and fallback paths and writes the decision record.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
SCRATCH=results/_throughput_probe

{
  while pgrep -f "09_cifar_pilot|03_generate_inrbench" > /dev/null; do sleep 30; done
  rm -rf "$SCRATCH"
  echo "--- clean throughput probe at the frozen config $(date) ---"
  $PY scripts/03_generate_inrbench.py --dataset cifar10 --protocol P-shared-det \
      --split val --limit 512 --steps 1000 --width 32 --layers 2 --batch 256 \
      --out-root "$SCRATCH" --tag probe
  $PY - <<'PYEOF'
import json
from pathlib import Path
import pandas as pd

probe = Path("results/_throughput_probe/cifar10/P-shared-det-probe/metadata.parquet")
per_fit = float(pd.read_parquet(probe)["wallclock_s"].median())
r_raw = 1.0 / per_fit
r = r_raw * 0.87  # sustained thermal-throttle derate measured on the MNIST chain (R7)

# Registered counts used 3 x 50k; a CIFAR protocol is actually 60k (45k train + 5k val + 10k
# test), so the corrected full-path count is 540k. Both are reported; the rule is applied to the
# corrected one because it is the honest number, and the decision is unchanged either way.
counts = {"registered_510k": 510_000, "corrected_540k": 540_000, "fallback_232k": 232_000}
hours = {k: v / r / 3600 for k, v in counts.items()}
decision = "full" if hours["corrected_540k"] <= 30 else (
    "fallback" if hours["fallback_232k"] <= 20 else "escalate-waiver")

record = {
    "rule": "R-CIFAR (docs/THINKING/G3-cifar-pilot.md)",
    "frozen_config": {"width": 32, "layers": 2, "steps": 1000, "lr": 1e-3},
    "median_s_per_fit": per_fit,
    "fits_per_s_raw": r_raw,
    "fits_per_s_derated": r,
    "projected_hours": hours,
    "thresholds": {"full_max_h": 30, "fallback_max_h": 20},
    "decision": decision,
}
Path("results/cifar_decision.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
PYEOF
  rm -rf "$SCRATCH"
} 2>&1 | tee -a results/cifar_pilot.log

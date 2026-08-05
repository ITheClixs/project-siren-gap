#!/usr/bin/env python3
"""Score the frozen S10 registration (docs/prereg/S10.md): the third arm, W12b.

Three intervals and three probability calls, appended once to docs/PREDICTION_OUTCOMES.csv, plus
the pre-committed interpretation branch of S10 section 4 -- which is selected by the number, not
chosen after seeing it.

Usage:
  .venv/bin/python scripts/56_score_s10.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTCOMES = ROOT / "docs" / "PREDICTION_OUTCOMES.csv"
LADDER = ROOT / "results" / "ladder" / "mnist"
DATE = "2026-08-05"
SEEDS = 5

REGISTERED = {
    "H-S10-1": ("f(W12b)", 0.60, (0.35, 0.80)),
    "H-S10-2": ("f(W12u) - f(W12b), the coordinate contribution", 0.26, (0.06, 0.48)),
    "H-S10-3": ("f(W12b) - f(W11a), the architecture contribution", 0.34, (0.10, 0.55)),
}
CALLS = {
    "P-S10-A": ("probability f(W12b) < f(W12u)", 0.85),
    "P-S10-B": ("probability f(W12b) > f(W11a)", 0.90),
    "P-S10-C": ("probability f(W12b) > 0.70 (architecture explains most of the step)", 0.35),
}


def branch(f: float) -> tuple[str, str]:
    """S10 section 4, evaluated mechanically."""
    if f >= 0.75:
        return ("architecture", "CLAIMS row 51's reading is WITHDRAWN rather than qualified: the "
                                "architecture, not the coordinates, carries most of the step.")
    if f <= 0.55:
        return ("coordinates", "Row 51 stands, the limitation closes, and the three-arm "
                               "decomposition is reported as measured.")
    return ("split", "Both contribute materially and neither dominates. The paper reports the "
                     "number and states the attribution is split, without picking the closer side.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()

    arms = {n: json.loads((LADDER / f"{n}.json").read_text())
            for n in ("W11", "W12", "W12u", "W12b")}
    f12b = arms["W12b"]["recovery_fraction"]
    f12u = arms["W12u"]["recovery_fraction"]
    f12 = arms["W12"]["recovery_fraction"]
    f11a = arms["W11"]["variants"]["W11a"]["recovery_fraction"]

    if len(arms["W12b"]["acc"]) != SEEDS:
        raise SystemExit(f"refusing to score at n={len(arms['W12b']['acc'])}; S10 fixes n={SEEDS}")
    if not arms["W12b"].get("raw_bias"):
        raise SystemExit("W12b.json is not a raw-bias run")

    obs = {"H-S10-1": f12b, "H-S10-2": f12u - f12b, "H-S10-3": f12b - f11a}
    rows, hits = [], 0
    print(f"S10 on mnist: f(W12b) = {f12b:.4f}\n")
    for k, v in obs.items():
        stmt, point, (lo, hi) = REGISTERED[k]
        hit = lo <= v <= hi
        hits += hit
        print(f"  {k}: registered {point} [{lo}, {hi}], observed {v:.4f} -> "
              f"{'HIT' if hit else 'MISS'}")
        rows.append({"date_scored": DATE, "prediction": f"{k} {stmt}", "kind": "interval",
                     "point": point, "lo80": lo, "hi80": hi, "observed": round(v, 4),
                     "verdict": "HIT" if hit else "MISS",
                     "abs_error": round(abs(v - point), 4), "brier": "", "note": ""})

    happened = {"P-S10-A": f12b < f12u, "P-S10-B": f12b > f11a, "P-S10-C": f12b > 0.70}
    for k, (stmt, p) in CALLS.items():
        o = bool(happened[k])
        print(f"  {k}: p={p}, resolved {o}, Brier {(p - o) ** 2:.4f}")
        rows.append({"date_scored": DATE, "prediction": f"{k} {stmt}", "kind": "probability",
                     "point": p, "lo80": "", "hi80": "", "observed": int(o), "verdict": "—",
                     "abs_error": "", "brier": round((p - o) ** 2, 4), "note": ""})

    tag, text = branch(f12b)
    decomp = {
        "architecture_within_the_graded_skeleton": round(f12b - f11a, 4),
        "coordinates_added_to_that_skeleton": round(f12 - f12b, 4),
        "layer_level_grading": round(f12 - f12u, 4),
    }
    print(f"\nS10 intervals: {hits}/3")
    print(f"pre-committed branch: {tag.upper()} -- {text}")
    print("decomposition of the W11a -> W12 step (0.265 -> 0.917):")
    for k, v in decomp.items():
        print(f"  {k:42s} {v:+.3f}")

    verdict = {"study": "S10", "prereg": "docs/prereg/S10.md",
               "f": {"W11a": f11a, "W12b": f12b, "W12u": f12u, "W12": f12},
               "acc_W12b": arms["W12b"]["mean"], "reader_params": arms["W12b"]["reader_params"],
               "intervals": f"{hits}/3", "branch": tag, "branch_text": text,
               "decomposition": decomp,
               "registered": {k: {"point": REGISTERED[k][1], "interval": list(REGISTERED[k][2]),
                                  "observed": round(obs[k], 4),
                                  "verdict": "HIT" if REGISTERED[k][2][0] <= obs[k]
                                  <= REGISTERED[k][2][1] else "MISS"} for k in REGISTERED},
               "calls": {k: {"p": CALLS[k][1], "resolved": bool(happened[k]),
                             "brier": round((CALLS[k][1] - happened[k]) ** 2, 4)} for k in CALLS}}

    if args.dry_run:
        return
    with OUTCOMES.open(newline="") as fh:
        already = {r["prediction"] for r in csv.DictReader(fh)}
    clash = sorted({r["prediction"] for r in rows} & already)
    if clash and not args.rescore:
        raise SystemExit(f"refusing to double-score: {len(clash)} rows already present "
                         f"(first: {clash[0]!r}); see CLAIMS row 54")
    with OUTCOMES.open(newline="") as fh:
        fieldnames = list(csv.DictReader(fh).fieldnames)
    with OUTCOMES.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writerows(rows)
    out = ROOT / "results" / "s10"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"\nappended {len(rows)} rows to {OUTCOMES}\nwrote {out / 'verdict.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score the frozen S4e registration (docs/prereg/S4e.md, aa5426a4245bd22f).

Evaluates the void conditions first, then the falsification criterion, then the nine
registered intervals and the probability call, and appends the rows to
docs/PREDICTION_OUTCOMES.csv. Nothing here selects a hypothesis: every threshold is fixed
by the committed registration.

Usage:
  .venv/bin/python scripts/27_score_s4e.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "results" / "s4e" / "s4e.json"
OUTCOMES = ROOT / "docs" / "PREDICTION_OUTCOMES.csv"
PREREG_HASH = "aa5426a4245bd22f"
DATE = "2026-07-29"

# prereg section 4
R_F_COUNTEREXAMPLE = 1e-5
KAPPA_SLACK = 20.0
RECOVERED = 0.05  # prereg section 5, P-S4e-7


def by_width(rows: list[dict]) -> dict[int, dict]:
    return {r["width"]: r for r in rows}


def check_void(arms: dict) -> list[str]:
    """Prereg section 4 void conditions. Any hit means the run is reported as void."""
    problems = []
    for r in arms.get("planted", []):
        if r["R_theta_max"] > 1e-5:
            problems.append(f"planted w={r['width']} max R_theta {r['R_theta_max']:.2e} > 1e-5")
    for r in arms.get("null", []):
        if r["R_theta_median"] < 0.15:
            problems.append(f"null w={r['width']} median {r['R_theta_median']:.3f} < 0.15")
    warm = arms.get("warmstart", [])
    if warm and all(r["recovered_frac"] == 0.0 for r in warm):
        problems.append("warmstart recovered nothing at any width or epsilon")
    return problems


def falsification(arms: dict) -> dict:
    """Prereg section 4: R_f < 1e-5 AND R_theta > 20 kappa R_f, planted passing at that width."""
    kappa = {r["width"]: r["ladder"][0]["kappa_median"] for r in arms.get("sensitivity", [])}
    planted_ok = {r["width"]: r["R_theta_max"] <= 1e-5 for r in arms.get("planted", [])}
    hits = []
    for row in arms.get("teacher", []):
        w = row["width"]
        k = kappa.get(w)
        for i, (rf, rt) in enumerate(zip(row["R_f"], row["R_theta"])):
            if rf < R_F_COUNTEREXAMPLE and k is not None:
                if rt > KAPPA_SLACK * k * rf and planted_ok.get(w, False):
                    hits.append({"width": w, "student": i, "R_f": rf, "R_theta": rt,
                                 "kappa": k, "threshold": KAPPA_SLACK * k * rf})
    return {"criterion_met": bool(hits), "candidates": hits,
            "n_students_below_R_f_threshold": sum(
                int(rf < R_F_COUNTEREXAMPLE) for row in arms.get("teacher", []) for rf in row["R_f"]
            )}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    report = json.loads(RESULT.read_text())
    arms = report["arms"]

    void = check_void(arms)
    print("VOID CONDITIONS:", "none" if not void else "")
    for v in void:
        print("  !!", v)
    if void:
        print("\nRun is VOID by the frozen registration. Not scoring.")
        return

    fals = falsification(arms)
    print(f"\nFALSIFICATION CRITERION: {'MET' if fals['criterion_met'] else 'not met'} "
          f"({fals['n_students_below_R_f_threshold']} students reached R_f < {R_F_COUNTEREXAMPLE:g})")
    for c in fals["candidates"][:5]:
        print(f"  candidate w={c['width']} student {c['student']}: "
              f"R_f {c['R_f']:.2e}, R_theta {c['R_theta']:.3e} > {c['threshold']:.2e}")

    planted = by_width(arms["planted"])
    sens = by_width(arms["sensitivity"])
    teach = by_width(arms["teacher"])
    warm = {(r["width"], r["eps_start"]): r for r in arms["warmstart"]}

    w2 = teach[2]
    rf2, rt2 = np.array(w2["R_f"]), np.array(w2["R_theta"])
    best = int(np.argmin(rf2))

    rows = [
        ("P-S4e-1", "planted control max R_theta worst width", 5e-8, 1e-9, 1e-5,
         max(r["R_theta_max"] for r in arms["planted"]), ""),
        ("P-S4e-2", "kappa median local condition number at w=32", 0.006, 0.003, 0.012,
         sens[32]["ladder"][0]["kappa_median"], "eps=1e-4 rung of the ladder"),
        ("P-S4e-3", "warmstart recovery fraction eps=1e-4 w=2", 0.85, 0.55, 1.00,
         warm[(2, 1e-4)]["recovered_frac"], ""),
        ("P-S4e-4", "warmstart recovery fraction eps=1e-4 w=32", 0.00, 0.00, 0.15,
         warm[(32, 1e-4)]["recovered_frac"], ""),
        ("P-S4e-5", "best R_f over independent students w=2", 3e-4, 1e-6, 3e-3,
         float(rf2.min()), ""),
        ("P-S4e-6", "R_theta at the best-R_f independent student w=2", 0.35, 0.05, 0.65,
         float(rt2[best]), ""),
        ("P-S4e-7", "independent-student recovery rate R_theta<0.05 at w=2", 0.02, 0.00, 0.15,
         float((rt2 < RECOVERED).mean()), ""),
    ]
    if "production" in arms:
        p = arms["production"]
        rows += [
            ("P-S4e-8", "production median R_theta same-image pairs w=32", 0.45, 0.25, 0.70,
             p["R_theta_median"], p["dataset"]),
            ("P-S4e-9", "production median R_theta minus different-image null", 0.00, -0.10, 0.10,
             p["R_theta_median"] - p["R_theta_null_median"], ""),
        ]

    out = []
    for pid, desc, point, lo, hi, obs, note in rows:
        hit = lo <= obs <= hi
        out.append({
            "date_scored": DATE, "prediction": f"{pid} {desc} (pilot-informed)",
            "kind": "interval", "point": point, "lo80": lo, "hi80": hi,
            "observed": round(float(obs), 6), "verdict": "HIT" if hit else "MISS",
            "abs_error": round(abs(float(obs) - point), 6), "brier": "", "note": note,
        })
    happened = fals["criterion_met"]
    out.append({
        "date_scored": DATE,
        "prediction": "P-S4e-C probability the falsification criterion is met at some width",
        "kind": "probability", "point": 0.15, "lo80": "", "hi80": "",
        "observed": int(happened), "verdict": "—", "abs_error": "",
        "brier": round((0.15 - float(happened)) ** 2, 4),
        "note": "conjecture 6.5 dies if 1" ,
    })

    n_iv = sum(1 for r in out if r["kind"] == "interval")
    n_hit = sum(1 for r in out if r["verdict"] == "HIT")
    print(f"\nS4e: {n_hit}/{n_iv} intervals hit")
    for r in out:
        print(f"  {r['prediction'].split()[0]:9s} obs={r['observed']:>12} {r['verdict']}")

    if args.dry_run:
        return
    fieldnames = list(csv.DictReader(OUTCOMES.open()).fieldnames)
    with OUTCOMES.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writerows(out)
    (ROOT / "results" / "s4e" / "verdict.json").write_text(json.dumps({
        "prereg": f"docs/prereg/S4e.md ({PREREG_HASH})",
        "void_conditions": void,
        "falsification": fals,
        "intervals_hit": f"{n_hit}/{n_iv}",
    }, indent=2))
    print(f"\nappended {len(out)} rows to {OUTCOMES}")


if __name__ == "__main__":
    main()

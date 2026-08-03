#!/usr/bin/env python3
"""Score S6, the orbit-only intervention, against its frozen registration (docs/prereg/S6.md).

Arm (i) was declared exposed in the registration and carries no ledger rows; it is reported as a
measurement. Arms (ii), (iii) and (iv) carry H-S6-1..6 and the two probability calls.

Usage:
  .venv/bin/python scripts/43_score_s6.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S6 = ROOT / "results" / "s6"

# frozen in docs/prereg/S6.md, section 4
REGISTERED = {
    "H-S6-1": ("Delta_sym, identity permutation, B=0", 10.0, (2.0, 30.0)),
    "H-S6-2": ("Delta_sym, identity permutation, B=3", 15.0, (3.0, 35.0)),
    "H-S6-3": ("Delta_sym(permuted,B=3) - Delta_sym(identity,B=3)", 64.0, (45.0, 76.0)),
    "H-S6-4": ("W11a recovery of Delta_sym at B=3", 0.60, (0.30, 0.88)),
    "H-S6-5": ("|W11b scattered - unscattered| (pts)", 0.3, (0.0, 1.5)),
    "H-S6-6": ("raw accuracy on scattered P-random, B=3", 13.0, (11.0, 15.5)),
}
PROB = {"P-S6-A": 0.80, "P-S6-B": 0.85}


def load(name: str) -> dict | None:
    p = S6 / f"orbit_mnist_{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def verdict(v: float, lo: float, hi: float) -> str:
    return "HIT" if lo <= v <= hi else "MISS"


def main() -> None:
    perm, noperm = load("perm"), load("noperm")
    equi, prand = load("equivariant"), load("prandom")
    if perm is None:
        print("S6: arm (i) not present (results/s6/orbit_mnist_perm.json)")
        return

    def dsym(run: dict, b: str) -> float | None:
        cell = run["by_winding"].get(b)
        return None if cell is None else cell["delta_sym"]

    observed: dict[str, float] = {}
    if noperm:
        if dsym(noperm, "0") is not None:
            observed["H-S6-1"] = dsym(noperm, "0")
        if dsym(noperm, "3") is not None and dsym(perm, "3") is not None:
            observed["H-S6-2"] = dsym(noperm, "3")
            observed["H-S6-3"] = dsym(perm, "3") - dsym(noperm, "3")
    if equi:
        for b, cell in equi["by_winding"].items():
            tr = cell["treatments"]
            if "W11a" in tr:
                observed["H-S6-4"] = tr["W11a"]["recovered_fraction"]
            if "W11b" in tr:
                base = ROOT / "results" / "ladder" / "mnist" / "W11_shareddet.json"
                if base.exists():
                    un = json.loads(base.read_text())["variants"]["W11b"]["mean"]
                    observed["H-S6-5"] = abs(tr["W11b"]["mean"] - un)
    if prand:
        cell = prand["by_winding"].get("3")
        if cell:
            observed["H-S6-6"] = cell["treatments"]["raw"]["mean"]

    report: dict = {
        "study": "S6", "prereg": "docs/prereg/S6.md",
        "note": "arm (i) was declared exposed before freezing and carries no ledger rows",
        "arms_present": {"perm": True, "noperm": bool(noperm),
                         "equivariant": bool(equi), "prandom": bool(prand)},
        "measurement": {
            "baseline_acc": perm["baseline"]["mean"],
            "delta_sym_permuted": {b: c["delta_sym"] for b, c in perm["by_winding"].items()},
            "recovery_permuted": {
                b: {k: v["recovered_fraction"] for k, v in c["treatments"].items()}
                for b, c in perm["by_winding"].items()
            },
            "max_functional_gap": max(c["functional_gap"] for c in perm["by_winding"].values()),
        },
        "registered": {},
    }
    if noperm:
        report["measurement"]["delta_sym_identity_perm"] = {
            b: c["delta_sym"] for b, c in noperm["by_winding"].items()
        }

    for k, v in observed.items():
        stmt, point, (lo, hi) = REGISTERED[k]
        report["registered"][k] = {
            "statement": stmt, "point": point, "interval": [lo, hi],
            "observed": v, "verdict": verdict(v, lo, hi),
        }
        print(f"{k}: registered {point} [{lo}, {hi}], observed {v:.3f} -> {verdict(v, lo, hi)}")

    calls: dict = {}
    if noperm and dsym(noperm, "3") is not None and dsym(perm, "3") is not None:
        truth = dsym(noperm, "3") < 0.5 * dsym(perm, "3")
        calls["P-S6-A"] = {
            "statement": "the permutation part dominates: Delta_sym(identity,B=3) < half of "
                         "Delta_sym(permuted,B=3)",
            "p": PROB["P-S6-A"], "resolved": bool(truth),
            "brier": (PROB["P-S6-A"] - float(truth)) ** 2,
        }
    if prand:
        cell = prand["by_winding"].get("3")
        if cell:
            cost = prand["baseline"]["mean"] - cell["treatments"]["raw"]["mean"]
            truth = cost < 2.0
            calls["P-S6-B"] = {
                "statement": "adding scatter to P-random costs less than 2 points",
                "observed_cost_pts": cost, "p": PROB["P-S6-B"], "resolved": bool(truth),
                "brier": (PROB["P-S6-B"] - float(truth)) ** 2,
            }
    report["probability_calls"] = calls
    for k, c in calls.items():
        print(f"{k}: p={c['p']}, resolved {c['resolved']}, Brier {c['brier']:.4f}")

    hits = sum(1 for r in report["registered"].values() if r["verdict"] == "HIT")
    report["score"] = f"{hits}/{len(report['registered'])}"
    print(f"\nS6 intervals: {report['score']}")

    (S6 / "verdict.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {S6 / 'verdict.json'}")


if __name__ == "__main__":
    main()

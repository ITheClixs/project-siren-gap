#!/usr/bin/env python3
"""Score the frozen S11 registration (docs/prereg/S11.md).

Two halves, scored as they land: the 2x2 completion on MNIST, and W12 on the three corpora it
was not designed on. Rows are appended once, per the guard of CLAIMS row 54, so running this
after the first half and again after the second is safe.

Usage:
  .venv/bin/python scripts/58_score_s11.py [--dry-run]
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
DATE = "2026-08-05"
SEEDS = 5
CALIGN = {"mnist": 0.628, "fashionmnist": 0.664, "cifar10": 0.324, "cifar10gray": 0.324}

REGISTERED = {
    "H-S11-1": ("f(W12ub), MNIST", 0.55, (0.30, 0.78)),
    "H-S11-2": ("the 2x2 interaction of grading and phasor lift", 0.00, (-0.25, 0.25)),
    "H-S11-3": ("s(W12), FashionMNIST", 0.85, (0.60, 0.95)),
    "H-S11-4": ("s(W12), luminance CIFAR-10", 0.60, (0.30, 0.85)),
    "H-S11-5": ("s(W12), RGB CIFAR-10", 0.60, (0.30, 0.85)),
}


def rf(path: Path) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if len(d.get("acc", [])) != SEEDS:
        raise SystemExit(f"{path.name} has {len(d.get('acc', []))} seeds; S11 fixes n={SEEDS}")
    return float(d["recovery_fraction"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    L = ROOT / "results" / "ladder" / "mnist"
    w11a = json.loads((L / "W11.json").read_text())["variants"]["W11a"]["recovery_fraction"]
    arms = {n: rf(L / f"{n}.json") for n in ("W12", "W12u", "W12b", "W12ub")}
    cross = {ds: rf(ROOT / "results" / "ladder" / ds / "W12.json")
             for ds in ("fashionmnist", "cifar10gray", "cifar10")}

    obs: dict[str, float] = {}
    decomp = {}
    if arms["W12ub"] is not None:
        ub, u, b, w = arms["W12ub"], arms["W12u"], arms["W12b"], arms["W12"]
        obs["H-S11-1"] = ub
        obs["H-S11-2"] = (w - b) - (u - ub)
        decomp = {
            "skeleton_W11a_to_W12ub": round(ub - w11a, 4),
            "phasor_lift_ungraded": round(u - ub, 4),
            "phasor_lift_graded": round(w - b, 4),
            "grading_raw_bias": round(b - ub, 4),
            "grading_phasor": round(w - u, 4),
            "interaction": round((w - b) - (u - ub), 4),
            "additive_sum": round(w11a + (ub - w11a) + (u - ub) + (w - u), 4),
        }
    for key, ds in (("H-S11-3", "fashionmnist"), ("H-S11-4", "cifar10gray"),
                    ("H-S11-5", "cifar10")):
        if cross[ds] is not None:
            obs[key] = cross[ds]

    rows, hits = [], 0
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

    # probability calls, decidable only when their inputs exist
    calls = {}
    if all(cross.values()) and arms["W12"] is not None:
        beats = all(cross[ds] > CALIGN[ds] for ds in cross) and arms["W12"] > CALIGN["mnist"]
        calls["P-S11-A"] = ("probability s(W12) beats c_align on every dataset", 0.80, beats)
        calls["P-S11-B"] = ("probability s(W12) on RGB CIFAR-10 exceeds W10's 0.534", 0.55,
                            cross["cifar10"] > 0.534)
    if "H-S11-2" in obs:
        calls["P-S11-C"] = ("probability the 2x2 interaction is outside [-0.10, +0.10]", 0.30,
                            not (-0.10 <= obs["H-S11-2"] <= 0.10))
    for k, (stmt, p, happened) in calls.items():
        print(f"  {k}: p={p}, resolved {bool(happened)}, Brier {(p - happened) ** 2:.4f}")
        rows.append({"date_scored": DATE, "prediction": f"{k} {stmt}", "kind": "probability",
                     "point": p, "lo80": "", "hi80": "", "observed": int(bool(happened)),
                     "verdict": "—", "abs_error": "",
                     "brier": round((p - happened) ** 2, 4), "note": ""})

    if decomp:
        print("\nthe 2x2, additively:")
        for k, v in decomp.items():
            print(f"  {k:26s} {v:+.4f}")

    missing = [k for k in REGISTERED if k not in obs]
    print(f"\nS11 so far: {hits}/{len(obs)} intervals; still owed: {missing or 'none'}")

    if args.dry_run:
        return
    with OUTCOMES.open(newline="") as fh:
        already = {r["prediction"] for r in csv.DictReader(fh)}
    rows = [r for r in rows if r["prediction"] not in already]
    if not rows:
        print("nothing new to append")
    else:
        with OUTCOMES.open(newline="") as fh:
            fieldnames = list(csv.DictReader(fh).fieldnames)
        with OUTCOMES.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerows(rows)
        print(f"appended {len(rows)} rows to {OUTCOMES}")
    out = ROOT / "results" / "s11"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(
        {"study": "S11", "prereg": "docs/prereg/S11.md", "f": {**arms, "W11a": w11a},
         "cross_dataset": cross, "decomposition": decomp,
         "intervals": f"{hits}/{len(obs)}", "still_owed": missing}, indent=2))
    print(f"wrote {out / 'verdict.json'}")


if __name__ == "__main__":
    main()

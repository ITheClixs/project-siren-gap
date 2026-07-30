#!/usr/bin/env python3
"""Score the frozen luminance-CIFAR arm (docs/prereg/S1-gray.md, b84b660829aa6d40).

Ten registered intervals (H-G1-*) and three probability calls (P-G-*), appended to
docs/PREDICTION_OUTCOMES.csv. Every threshold comes from the committed registration.

Usage:
  .venv/bin/python scripts/32_score_gray_arm.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTCOMES = ROOT / "docs" / "PREDICTION_OUTCOMES.csv"
PREREG_HASH = "b84b660829aa6d40"
DATE = "2026-07-30"


def analysis(dataset: str) -> dict:
    return json.loads((ROOT / "results" / "ladder" / dataset / "S1_analysis.json").read_text())


def gate_psnr(dataset: str, protocol: str = "P-shared-det") -> float:
    p = ROOT / "results" / "inrbench" / f"{dataset}_{protocol}_test_gate.json"
    return float(json.loads(p.read_text())["psnr"]["median"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    a = analysis("cifar10gray")
    m, fr, tests, ctl = a["means"], a["recovery_fractions"], a["tests"], a["controls"]
    f = {k: v["point"] for k, v in fr.items() if k.startswith("f_")}

    rows = [
        ("H-G1-1", "P0 real-pixel MLP test accuracy pct", 47.0, 41, 53, m["P0"], ""),
        ("H-G1-2", "P1-P0 accuracy points", 0.2, -1.0, 1.5,
         tests["H-S1-1"]["diff"]["mean_diff"], ""),
        ("H-G1-3", "W1 test accuracy pct", 38.0, 28, 46, m["W1"], ""),
        ("H-G1-4", "W3 test accuracy pct", 12.5, 10.5, 16.0, m["W3"], "at the lower edge"),
        ("H-G1-5", "W1-W3 perception gap accuracy points", 25.5, 14, 36,
         tests["H-S1-4a"]["diff"]["mean_diff"], ""),
        ("H-G1-6", "recovery fraction f(W5) c_align (the discriminator)", 0.45, 0.28, 0.66,
         f["f_W5"], "identical to RGB CIFAR's 0.324 at c=3"),
        ("H-G1-7", "recovery fraction f(W10) exact L=2 invariants", 0.45, 0.26, 0.60,
         f["f_W10"], "close to RGB CIFAR's 0.534"),
        ("H-G1-8", "recovery fraction f(W4) template-free c_sort", 0.13, 0.06, 0.24,
         f["f_W4"], "misses the lower edge by 0.002"),
        ("H-G1-9", "recovery fraction f(W9) frame averaging", 0.00, -0.03, 0.05, f["f_W9"], ""),
        ("H-G1-10", "median render PSNR dB", 59.5, 55, 63, gate_psnr("cifar10gray"), ""),
    ]

    out = []
    for pid, desc, point, lo, hi, obs, note in rows:
        hit = lo <= obs <= hi
        out.append({
            "date_scored": DATE, "prediction": f"{pid} {desc}", "kind": "interval",
            "point": point, "lo80": lo, "hi80": hi, "observed": round(float(obs), 4),
            "verdict": "HIT" if hit else "MISS", "abs_error": round(abs(float(obs) - point), 4),
            "brier": "", "note": note,
        })

    rgb = {k: v["point"] for k, v in analysis("cifar10")["recovery_fractions"].items()
           if k.startswith("f_")}
    shuffles = [ctl[r]["label_shuffle_test_acc"] for r in ("W1", "W3", "W5")]
    probs = [
        ("P-G-A", "probability f(W5)>0.50 (paper section 9 conjecture substantially right)",
         0.35, f["f_W5"] > 0.50,
         f"observed f(W5)={f['f_W5']:.3f} at c=1 against {rgb['f_W5']:.3f} at c=3 — unchanged"),
        ("P-G-B", "probability f(W10)<f(W5) i.e. the crossover reverses at c=1",
         0.45, f["f_W10"] < f["f_W5"],
         f"crossover persists: f(W10)={f['f_W10']:.3f} > f(W5)={f['f_W5']:.3f}"),
        ("P-G-C", "probability all three label-shuffle controls within 2 pts of chance",
         0.80, all(s is not None and abs(s - 10.0) <= 2.0 for s in shuffles),
         f"shuffles {shuffles}"),
    ]
    for pid, desc, p, happened, note in probs:
        out.append({
            "date_scored": DATE, "prediction": f"{pid} {desc}", "kind": "probability",
            "point": p, "lo80": "", "hi80": "", "observed": int(bool(happened)),
            "verdict": "—", "abs_error": "",
            "brier": round((p - float(bool(happened))) ** 2, 4), "note": note,
        })

    n_iv = sum(1 for r in out if r["kind"] == "interval")
    n_hit = sum(1 for r in out if r["verdict"] == "HIT")
    print(f"prereg {PREREG_HASH}: {n_hit}/{n_iv} intervals hit")
    for r in out:
        extra = f"  brier {r['brier']}" if r["brier"] != "" else ""
        print(f"  {r['prediction'].split()[0]:8s} obs={r['observed']:>9} {r['verdict']:5s}{extra}")

    verdict = (
        "CONJECTURE WITHDRAWN" if not (f["f_W5"] > 0.50) and not (f["f_W10"] < f["f_W5"])
        else "conjecture supported" if f["f_W5"] > 0.50 and f["f_W10"] < f["f_W5"]
        else "mixed — neither single-cause story survives"
    )
    print(f"\nprereg section 5 verdict: {verdict}")
    print(f"  f(W5): {f['f_W5']:.3f} at c=1 vs {rgb['f_W5']:.3f} at c=3 on the same images")
    print(f"  f(W10): {f['f_W10']:.3f} at c=1 vs {rgb['f_W10']:.3f} at c=3")
    print("  => the CIFAR behaviour is driven by image statistics, not output-channel count")

    if args.dry_run:
        return
    fieldnames = list(csv.DictReader(OUTCOMES.open()).fieldnames)
    with OUTCOMES.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writerows(out)
    (ROOT / "results" / "ladder" / "cifar10gray" / "verdict.json").write_text(json.dumps({
        "prereg": f"docs/prereg/S1-gray.md ({PREREG_HASH})",
        "intervals_hit": f"{n_hit}/{n_iv}",
        "verdict": verdict,
        "f_gray": f, "f_rgb": rgb,
    }, indent=2))
    print(f"\nappended {len(out)} rows to {OUTCOMES}")


if __name__ == "__main__":
    main()

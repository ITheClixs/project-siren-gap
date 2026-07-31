#!/usr/bin/env python3
"""Score the frozen W11 registration (docs/prereg/S1-w11.md, e3bbc081a5810956).

Five intervals (H-W11-*) and three probability calls (P-W11-*), appended to
docs/PREDICTION_OUTCOMES.csv. Also prints the verdict that prereg §5 pre-committed for each
outcome, so the write-up cannot drift from what was registered.

Usage:
  .venv/bin/python scripts/34_score_w11.py [--dataset mnist] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTCOMES = ROOT / "docs" / "PREDICTION_OUTCOMES.csv"
PREREG_HASH = "e3bbc081a5810956"
DATE = "2026-07-31"


def paired_f(acc: list[float], w3: list[float], w1: list[float]) -> float:
    n = min(len(acc), len(w3), len(w1))
    a, b, c = (np.array(x[:n]) for x in (acc, w3, w1))
    return float(((a - b) / (c - b)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ladder = ROOT / "results" / "ladder" / args.dataset
    w11 = json.loads((ladder / "W11.json").read_text())["variants"]
    ana = json.loads((ladder / "S1_analysis.json").read_text())
    cells = {r: json.loads((ladder / f"{r}.json").read_text())["acc"] for r in ("W1", "W3", "W10")}

    a, b = w11["W11a"], w11["W11b"]
    f_a, f_b = a["recovery_fraction"], b["recovery_fraction"]
    f_w10 = ana["recovery_fractions"]["f_W10"]["point"]
    f_w5 = ana["recovery_fractions"]["f_W5"]["point"]
    f_w4 = ana["recovery_fractions"]["f_W4"]["point"]
    # paired difference on common seeds, which is what H-W11-5 registered
    d_pool = paired_f(b["acc"], cells["W3"], cells["W1"]) - paired_f(
        cells["W10"], cells["W3"], cells["W1"]
    )

    rows = [
        ("H-W11-1", "recovery fraction f(W11a)", 0.26, 0.14, 0.42, f_a,
         f"vs c_sort {f_w4:.3f}, c_align {f_w5:.3f}"),
        ("H-W11-2", "recovery fraction f(W11b)", 0.44, 0.30, 0.62, f_b,
         f"vs W10 {f_w10:.3f}"),
        ("H-W11-3", "W11a test accuracy pct", 34.5, 25, 48, a["mean"],
         f"{a['reader_params']:,} params at width {a['width']}"),
        ("H-W11-4", "W11b test accuracy pct", 49.3, 38, 64, b["mean"],
         f"{b['reader_params']:,} params at width {b['width']}"),
        ("H-W11-5", "f(W11b)-f(W10) the pooling question", 0.17, 0.02, 0.34, d_pool, ""),
    ]
    out = []
    for pid, desc, point, lo, hi, obs, note in rows:
        hit = lo <= obs <= hi
        out.append({
            "date_scored": DATE, "prediction": f"{pid} {desc} (pilot-informed)",
            "kind": "interval", "point": point, "lo80": lo, "hi80": hi,
            "observed": round(float(obs), 4), "verdict": "HIT" if hit else "MISS",
            "abs_error": round(abs(float(obs) - point), 4), "brier": "", "note": note,
        })

    probs = [
        ("P-W11-A", "probability f(W11a)<f(W5) i.e. frame choice beats reader architecture",
         0.75, f_a < f_w5, f"f(W11a)={f_a:.3f} vs f(W5)={f_w5:.3f}"),
        ("P-W11-B", "probability f(W11b)>f(W10) i.e. learned pooling beats eigenvalue spectra",
         0.70, f_b > f_w10, f"f(W11b)={f_b:.3f} vs f(W10)={f_w10:.3f}"),
        ("P-W11-C", "probability f(W11b)>f(W5) i.e. invariant reader overtakes alignment",
         0.30, f_b > f_w5, f"f(W11b)={f_b:.3f} vs f(W5)={f_w5:.3f}"),
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
    print(f"prereg {PREREG_HASH} on {args.dataset}: {n_hit}/{n_iv} intervals hit")
    for r in out:
        extra = f"  brier {r['brier']}" if r["brier"] != "" else ""
        print(f"  {r['prediction'].split()[0]:9s} obs={r['observed']:>9} {r['verdict']:5s}{extra}")

    print("\nprereg section 5 verdicts:")
    if f_a < f_w5:
        print(f"  - W11a ({f_a:.3f}) is below c_align ({f_w5:.3f}): the paper's practical claim")
        print("    STANDS, bounded to one equivariant family at matched capacity on one corpus.")
    else:
        print(f"  - W11a ({f_a:.3f}) reaches or beats c_align ({f_w5:.3f}): the practical claim is")
        print("    WITHDRAWN. The decomposition results are unaffected; the recommendation dies.")
    if f_b > f_w10:
        print(f"  - W11b ({f_b:.3f}) beats W10 ({f_w10:.3f}) by {f_b - f_w10:+.3f}: pooling is part")
        print(f"    of the bottleneck; the residual against c_align ({f_w5 - f_b:+.3f}) is the")
        print("    invariants' incompleteness. OPEN_PROBLEMS #4 answered with a split.")
    else:
        print(f"  - W11b ({f_b:.3f}) does not beat W10 ({f_w10:.3f}): learned pooling buys nothing,")
        print("    the invariants themselves are the limit. OPEN_PROBLEMS #4 closes the other way.")

    if args.dry_run:
        return
    fieldnames = list(csv.DictReader(OUTCOMES.open()).fieldnames)
    with OUTCOMES.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writerows(out)
    (ladder / "W11_verdict.json").write_text(json.dumps({
        "prereg": f"docs/prereg/S1-w11.md ({PREREG_HASH})",
        "intervals_hit": f"{n_hit}/{n_iv}",
        "f": {"W11a": f_a, "W11b": f_b, "W4": f_w4, "W10": f_w10, "W5": f_w5},
        "pooling_gain": d_pool,
        "reader_params": {"W11a": a["reader_params"], "W11b": b["reader_params"],
                          "matched_mlp": 1873162},
    }, indent=2))
    print(f"\nappended {len(out)} rows to {OUTCOMES}")


if __name__ == "__main__":
    main()

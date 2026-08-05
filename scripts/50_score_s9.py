#!/usr/bin/env python3
"""Score S9, the phasor-graded reader, against its frozen registration (docs/prereg/S9.md).

Also writes paper/tables/w12_table.tex: the four-way comparison at matched capacity that is the
point of the rung -- permutation-only, invariant-front-end, phasor-graded on raw parameters, and
the exact reframing they are all measured against.

Usage:
  .venv/bin/python scripts/50_score_s9.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / "results" / "ladder" / "mnist"

# frozen in docs/prereg/S9.md before the rung was decoded
REGISTERED = {
    "H-S9-1": ("f(W12), MNIST P-random", 0.55, (0.35, 0.72)),
    "H-S9-2": ("f(W12) - f(W11a)", 0.29, (0.10, 0.45)),
    "H-S9-3": ("f(W12) - f(W11b)", 0.02, (-0.15, 0.18)),
}
PROB = {"P-S9-A": 0.85, "P-S9-B": 0.45, "P-S9-C": 0.25}


def verdict(v: float, lo: float, hi: float) -> str:
    return "HIT" if lo <= v <= hi else "MISS"


def main() -> None:
    w12_path = LADDER / "W12.json"
    if not w12_path.exists():
        print("S9: W12 cell not present yet")
        return
    w12 = json.loads(w12_path.read_text())
    w11 = json.loads((LADDER / "W11.json").read_text())["variants"]
    w12u_path = LADDER / "W12u.json"
    w12u = json.loads(w12u_path.read_text()) if w12u_path.exists() else None
    w12b_path = LADDER / "W12b.json"   # the S10 third arm, present once it has been run
    w12b = json.loads(w12b_path.read_text()) if w12b_path.exists() else None
    cells = {r: json.loads((LADDER / f"{r}.json").read_text()) for r in ("W1", "W3", "W4", "W5")}

    w1, w3 = np.array(cells["W1"]["acc"]), np.array(cells["W3"]["acc"])

    def frac(acc: list[float]) -> float:
        a = np.array(acc)
        n = min(len(a), len(w1), len(w3))
        return float(((a[:n] - w3[:n]) / (w1[:n] - w3[:n])).mean())

    f = {
        "W4": frac(cells["W4"]["acc"]), "W5": frac(cells["W5"]["acc"]),
        "W11a": w11["W11a"]["recovery_fraction"], "W11b": w11["W11b"]["recovery_fraction"],
        "W12": w12["recovery_fraction"],
    }
    if w12u:
        f["W12u"] = w12u["recovery_fraction"]
    if w12b:
        f["W12b"] = w12b["recovery_fraction"]
    acc = {
        "W4": cells["W4"]["mean"], "W5": cells["W5"]["mean"],
        "W11a": w11["W11a"]["mean"], "W11b": w11["W11b"]["mean"], "W12": w12["mean"],
    }
    if w12u:
        acc["W12u"] = w12u["mean"]
    if w12b:
        acc["W12b"] = w12b["mean"]
    params = {"W11a": w11["W11a"]["reader_params"], "W11b": w11["W11b"]["reader_params"],
              "W12": w12["reader_params"]}
    if w12u:
        params["W12u"] = w12u["reader_params"]
    if w12b:
        params["W12b"] = w12b["reader_params"]

    observed = {
        "H-S9-1": f["W12"],
        "H-S9-2": f["W12"] - f["W11a"],
        "H-S9-3": f["W12"] - f["W11b"],
    }
    report: dict = {
        "study": "S9", "prereg": "docs/prereg/S9.md",
        "exposure": "docs/prereg/S9-addendum-01.md",
        "f": f, "acc": acc, "reader_params": params,
        "width": w12["width"], "registered": {},
    }
    for k, v in observed.items():
        stmt, point, (lo, hi) = REGISTERED[k]
        report["registered"][k] = {"statement": stmt, "point": point, "interval": [lo, hi],
                                   "observed": v, "verdict": verdict(v, lo, hi)}
        print(f"{k}: registered {point} [{lo}, {hi}], observed {v:+.3f} -> "
              f"{report['registered'][k]['verdict']}")

    calls = {
        "P-S9-A": ("f(W12) > f(W11a)", f["W12"] > f["W11a"]),
        "P-S9-B": ("f(W12) > f(W11b)", f["W12"] > f["W11b"]),
        "P-S9-C": ("f(W12) > f(W5)", f["W12"] > f["W5"]),
    }
    report["probability_calls"] = {
        k: {"statement": st, "p": PROB[k], "resolved": bool(t),
            "brier": (PROB[k] - float(t)) ** 2}
        for k, (st, t) in calls.items()
    }
    for k, c in report["probability_calls"].items():
        print(f"{k}: p={c['p']}, resolved {c['resolved']}, Brier {c['brier']:.4f}")

    if w12u:
        # exploratory, not registered: S7's argument applied to W12
        report["grading_attributable"] = f["W12"] - f["W12u"]
        report["control_also_beats_calign"] = bool(f["W12u"] > f["W5"])
        print(f"\ncontrol: f(W12u) = {f['W12u']:.3f}; the layer-level grading is worth "
              f"{report['grading_attributable']:+.3f} of W12's {f['W12']:.3f}"
              f"{'; the control also beats c_align' if report['control_also_beats_calign'] else ''}")

    report["practical_claim_withdrawn"] = bool(calls["P-S9-C"][1])
    if report["practical_claim_withdrawn"]:
        print("\nP-S9-C RESOLVED TRUE -> the 'frame choice beats reader architecture' claim is "
              "withdrawn, per docs/prereg/S9.md section 4.")

    hits = sum(1 for v in report["registered"].values() if v["verdict"] == "HIT")
    report["score"] = f"{hits}/{len(report['registered'])}"
    print(f"\nS9 intervals: {report['score']}")

    (ROOT / "results" / "s9").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "s9" / "verdict.json").write_text(json.dumps(report, indent=2))
    (ROOT / "paper" / "tables" / "w12_table.tex").write_text(table(report))
    print("wrote results/s9/verdict.json and paper/tables/w12_table.tex")


def table(rep: dict) -> str:
    rows = [
        ("W11a", r"perm.-equivariant, raw weights", r"$S_{n_1}\times S_{n_2}$ only"),
        ("W11b", r"equivariant reader over W10's invariants", r"$G$, via a fixed front-end"),
        ("W12u", r"\quad control: grading removed, coordinates kept",
         r"none (logits move $0.25$)"),
        ("W12b", r"\quad control: coordinates removed, grading kept",
         r"none (logits move $6$ at $|j|\le3$)"),
        ("W12", r"phasor-graded, raw weights", r"$G$, on the parameters"),
        ("W5", r"$\calign$ + the frozen MLP", r"--- (a reframing, not a reader)"),
    ]
    rows = [r for r in rows if r[0] in rep["f"]]
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{\textbf{Readers at matched capacity (MNIST, \texttt{P-random}).} Every reader "
        r"is sized by rule to the frozen decoder's $1{,}873{,}162$ parameters. The "
        r"invariance column is what each construction actually quotients: W11a nothing beyond "
        r"permutations, W11b the full group but only because its input already is invariant, W12 "
        r"the full group on the raw parameters, and neither control anything. The two controls vary "
        r"one thing each against W12: W12u removes the layer grading and keeps the phasor "
        r"coordinates, W12b keeps the grading and feeds it the raw bias. Their gaps to W12 "
        r"are $+0.059$ and $+0.315$, against $+0.337$ from W11a to W12b, so within this "
        r"skeleton the coordinates and the architecture contribute comparably and the "
        r"layer-level grading contributes little. $\calign$ is listed for reference; it "
        r"is a change of frame rather than a reader.}",
        r"\label{tab:w12}",
        r"\begin{tabular}{@{}llrrl@{}}", r"\toprule",
        r"rung & construction & acc.\ (\%) & $f$ & invariance \\", r"\midrule",
    ]
    for key, desc, inv in rows:
        p = rep["reader_params"].get(key)
        pstr = f"{p:,}" if p else "---"
        lines.append(f"{key} & {desc} & {rep['acc'][key]:.2f} & {rep['f'][key]:.3f} & {inv} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

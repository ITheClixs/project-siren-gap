#!/usr/bin/env python3
"""Score S7: the matched non-invariant control for W10 (docs/prereg/S7.md).

Reads the ladder cells, computes the recovery fraction of the control against the frozen
registration, and writes paper/tables/control_table.tex.

Usage:
  .venv/bin/python scripts/41_score_s7.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sirengap.eval.stats import paired_summary  # noqa: E402

DATASETS = ("mnist", "cifar10")
PRETTY = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
NEEDED = ("W1", "W3", "W4", "W10", "W10c")

# frozen in docs/prereg/S7.md before any cell was decoded
REGISTERED = {
    "H-S7-1": ("f(W10c), MNIST", 0.14, (0.02, 0.28)),
    "H-S7-2": ("f(W10c), CIFAR-10", 0.22, (0.05, 0.42)),
    "H-S7-3": ("f(W10) - f(W10c), CIFAR-10", 0.31, (0.11, 0.48)),
}
FALSIFIER = 0.05  # f(W10c) >= f(W10) - 0.05 on CIFAR-10 voids the invariance reading


def cells(dataset: str) -> dict[str, dict] | None:
    d = ROOT / "results" / "ladder" / dataset
    out = {}
    for rung in NEEDED:
        p = d / f"{rung}.json"
        if not p.exists():
            return None
        out[rung] = json.loads(p.read_text())
    return out


def frac(c: dict[str, dict], rung: str) -> float:
    return (c[rung]["mean"] - c["W3"]["mean"]) / (c["W1"]["mean"] - c["W3"]["mean"])


def verdict(value: float, lo: float, hi: float) -> str:
    return "HIT" if lo <= value <= hi else "MISS"


def main() -> None:
    by_ds = {d: cells(d) for d in DATASETS}
    have = {d: c for d, c in by_ds.items() if c}
    if not have:
        print("S7: no W10c cells yet")
        return

    report: dict = {"study": "S7", "prereg": "docs/prereg/S7.md", "datasets": {}}
    for ds, c in have.items():
        f10, f10c, f4 = frac(c, "W10"), frac(c, "W10c"), frac(c, "W4")
        stats = paired_summary(np.array(c["W10"]["acc"]), np.array(c["W10c"]["acc"]))
        report["datasets"][ds] = {
            "acc": {r: c[r]["mean"] for r in NEEDED},
            "feature_dim": {r: c[r]["feature_dim"] for r in ("W10", "W10c")},
            "f_W4": f4, "f_W10": f10, "f_W10c": f10c,
            "delta_f": f10 - f10c,
            "paired_W10_vs_W10c": stats,
        }
        print(f"{ds}: f(W4)={f4:.3f}  f(W10)={f10:.3f}  f(W10c)={f10c:.3f}  "
              f"delta={f10 - f10c:+.3f}  D={c['W10']['feature_dim']}/{c['W10c']['feature_dim']}")

    scored = {}
    if "mnist" in have:
        scored["H-S7-1"] = report["datasets"]["mnist"]["f_W10c"]
    if "cifar10" in have:
        scored["H-S7-2"] = report["datasets"]["cifar10"]["f_W10c"]
        scored["H-S7-3"] = report["datasets"]["cifar10"]["delta_f"]
    report["registered"] = {
        k: {"statement": REGISTERED[k][0], "point": REGISTERED[k][1],
            "interval": list(REGISTERED[k][2]), "observed": v,
            "verdict": verdict(v, *REGISTERED[k][2])}
        for k, v in scored.items()
    }
    for k, r in report["registered"].items():
        print(f"  {k}: registered {r['point']} {r['interval']}, observed {r['observed']:.3f} "
              f"-> {r['verdict']}")

    if "cifar10" in have:
        d = report["datasets"]["cifar10"]
        report["falsifier_fired"] = bool(d["f_W10c"] >= d["f_W10"] - FALSIFIER)
        print(f"  falsifier (f(W10c) >= f(W10) - {FALSIFIER}): "
              f"{'FIRED' if report['falsifier_fired'] else 'not fired'}")
    if all(d in have for d in DATASETS):
        report["P-S7-A"] = all(report["datasets"][d]["delta_f"] > 0 for d in DATASETS)
    if "cifar10" in have:
        c = report["datasets"]["cifar10"]
        report["P-S7-B"] = bool(c["f_W10c"] <= c["f_W4"])

    out = ROOT / "results" / "s7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "control.json").write_text(json.dumps(report, indent=2))
    (ROOT / "paper" / "tables" / "control_table.tex").write_text(table(report))
    print(f"\nwrote {out / 'control.json'} and paper/tables/control_table.tex")


def table(report: dict) -> str:
    ds = [d for d in DATASETS if d in report["datasets"]]
    lines = [
        r"\begin{table}[t]", r"\centering\small",
        r"\caption{\textbf{The matched non-invariant control.} W10c emits the same monomials in "
        r"$(w,u)$ at the same trigonometric orders, pooled the same way, at the same dimension, "
        r"decoded by the same apparatus; only the parity classes differ, so it stays exactly "
        r"permutation-invariant and is not $\Dinf$-invariant. The gap between the two rows is the "
        r"part of the encoding's recovery attributable to the $\Dinf$ component.}",
        r"\label{tab:control}",
        r"\begin{tabular}{@{}l" + "rr" * len(ds) + r"@{}}", r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{PRETTY[d]}}}" for d in ds) + r" \\",
        "rung & " + " & ".join(r"acc.\ (\%) & $f$" for _ in ds) + r" \\", r"\midrule",
    ]
    for rung, label in (("W4", r"W4 $\csort$ (reference)"),
                        ("W10", "W10 exact invariants"),
                        ("W10c", "W10c matched control")):
        cells_ = []
        for d in ds:
            r = report["datasets"][d]
            cells_ += [f"{r['acc'][rung]:.2f}", f"{r['f_' + rung]:.3f}"]
        lines.append(f"{label} & " + " & ".join(cells_) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"$f(\mathrm{W10}) - f(\mathrm{W10c})$ & "
                 + " & ".join(rf"\multicolumn{{2}}{{c}}{{{report['datasets'][d]['delta_f']:.3f}}}"
                              for d in ds) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

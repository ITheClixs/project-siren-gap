"""Submission gate: every headline number in the paper, checked against its artifact.

Reads the result JSONs and the prediction ledger and asserts the figures quoted in
paper/tmlr-anon match them. Run before submitting; a failure means the paper and the
artifacts disagree, which is the one defect a reviewer can always verify.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "tmlr-anon"


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def mean(xs) -> float:
    return sum(xs) / len(xs)


def paper_text() -> str:
    parts = [(PAPER / "main.tex").read_text()]
    parts += [p.read_text() for p in sorted((PAPER / "sections").glob("*.tex"))]
    parts += [p.read_text() for p in sorted((PAPER / "tables").glob("*.tex"))]
    return "\n".join(parts)


def check(label: str, computed: float, quoted: float, tol: float = 0.011) -> bool:
    ok = abs(computed - quoted) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {label:52s} artifact {computed:10.4f}  paper {quoted:10.4f}")
    return ok


def appears(txt: str, needle: str) -> bool:
    flat = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", txt.replace("\n", " "))
    ok = needle in flat
    print(f"  {'ok ' if ok else 'FAIL'} quoted in paper: {needle}")
    return ok


if __name__ == "__main__":
    txt = paper_text()
    good = True

    print("\nladder anchors and readers (MNIST)")
    w11 = load("results/ladder/mnist/W11.json")
    good &= check("W1 shared-init anchor", w11["W1"], 94.36, 0.005)
    good &= check("W3 plain reader, independent fits", w11["W3"], 13.92, 0.005)
    good &= check("W11b P-random", w11["variants"]["W11b"]["mean"], 56.24)
    good &= check("W11b P-shared-det",
                  load("results/ladder/mnist/W11_shareddet.json")["variants"]["W11b"]["mean"], 84.81)
    w12r = mean(load("results/ladder/mnist/W12.json")["acc"]
                if isinstance(load("results/ladder/mnist/W12.json"), dict)
                else load("results/ladder/mnist/W12.json"))
    good &= check("W12 P-random", w12r, 87.64)
    w12s = load("results/ladder/mnist/W12_shareddet.json")
    w12s = mean(w12s["acc"] if isinstance(w12s, dict) else w12s)
    good &= check("W12 P-shared-det", w12s, 95.46)
    good &= check("W12 residual (shared - random)", w12s - w12r, 7.82)

    print("\nthe residual table of the discussion")
    good &= check("plain reader residual", w11["W1"] - w11["W3"], 80.44)
    good &= check("W11b residual",
                  load("results/ladder/mnist/W11_shareddet.json")["variants"]["W11b"]["mean"]
                  - w11["variants"]["W11b"]["mean"], 28.57)
    sg_r = load("results/s19/published/scalegmn_P-random-fold_seed0.summary.json")
    sg_s = load("results/s19/published/scalegmn_P-shared-det-fold_seed0.summary.json")
    good &= check("ScaleGMN P-random", sg_r["test_acc"], 94.76, 0.005)
    good &= check("ScaleGMN P-shared-det", sg_s["test_acc"], 96.36, 0.005)
    good &= check("ScaleGMN residual", sg_s["test_acc"] - sg_r["test_acc"], 1.60)
    good &= check("ScaleGMN s on P-random", sg_r["s"], 1.005, 0.0005)
    good &= check("ScaleGMN parameters (millions)", sg_r["reader_params"] / 1e6, 1.772042, 1e-6)

    f16 = load("results/s16/function_nuisance_mnist.json")["by_probes"]
    good &= check("function access K=64, P-random", f16["64"]["P-random"]["mean"], 95.34)
    good &= check("function access K=64, P-shared-det", f16["64"]["P-shared-det"]["mean"], 95.09)
    good &= check("function access K=64 difference", f16["64"]["difference"]["mean"], -0.25)

    print("\ncorrected W11a cell (S19 addendum 01)")
    fx = load("results/ladder/mnist/W11a_fixednorm.json")
    v = fx["variants"]["W11a"] if "variants" in fx else fx
    acc = v.get("mean", mean(v["acc"]))
    good &= check("W11a pooled-statistics accuracy", acc, 34.88, 0.02)

    print("\nprediction ledger")
    import csv
    rows = [r for r in csv.DictReader((ROOT / "docs/PREDICTION_OUTCOMES.csv").open())
            if r["kind"].strip() == "interval"]
    hits = sum(1 for r in rows if r["verdict"].strip().upper() == "HIT")
    good &= check("ledger intervals", len(rows), 138, 0)
    good &= check("ledger hits", hits, 103, 0)
    s19 = [r for r in rows if r["prediction"].startswith("H-S19")]
    good &= check("S19 intervals", len(s19), 16, 0)
    good &= check("S19 hits", sum(1 for r in s19 if r["verdict"].strip().upper() == "HIT"), 12, 0)

    print("\nstrings the paper must contain")
    for needle in ["$1.60$", "$96.36\\%$", "$94.76\\%$", "$80.44$", "$28.57$", "QG-7"]:
        good &= appears(txt, needle)

    print("\n" + ("ALL CHECKS PASS" if good else "SOME CHECKS FAILED"))
    sys.exit(0 if good else 1)

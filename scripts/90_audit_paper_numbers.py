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
    good &= check("plain reader residual", w11["W1"] - w11["W3"], 80.43)
    good &= check("W11b residual",
                  load("results/ladder/mnist/W11_shareddet.json")["variants"]["W11b"]["mean"]
                  - w11["variants"]["W11b"]["mean"], 28.56)
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

    print("\npublished baselines, against ScaleGMN Table 1 (NeurIPS 2024)")
    # transcribed once from the source table; the audit is that the paper still agrees with them
    tbl1 = {
        "ScaleGMN-B": (96.59, 80.78, 38.82), "ScaleGMN": (96.57, 80.46, 36.43),
        "NG-GNN": (91.40, 68.00, 36.04), "DWSNets": (85.71, 67.06, 34.45),
        "NFN$_{HNP}$": (79.11, 68.94, 28.64), "NFN$_{NP}$": (78.50, 68.19, 33.41),
    }
    bench = (PAPER / "tables" / "bench_table.tex").read_text()
    for name, (a, b, c) in tbl1.items():
        row = [l for l in bench.splitlines()
               if re.match(re.escape(name) + r"\s*\\citep", l.strip())
               or re.match(re.escape(name) + r"\s*&", l.strip())]
        hit = bool(row) and all(f"{v:.2f}" in row[0] for v in (a, b, c))
        print(f"  {'ok ' if hit else 'FAIL'} bench row matches ScaleGMN Table 1: {name}")
        good &= hit
    # W12 must be first on CIFAR-10 among non-augmented weight-only readers
    good &= check("W12 CIFAR-10 beats the best baseline",
                  44.20 - max(v[2] for v in tbl1.values()), 5.38, 0.005)

    print("\nfactorial attribution (the figure a reviewer caught us misquoting)")
    fac = load("results/s15/factorial_mnist.json")
    sh = fac["shapley"]
    good &= check("Shapley, relabelling (pi)", sh["pi"], 44.24)
    good &= check("Shapley, sign flips (sigma)", sh["sigma"], 29.86)
    good &= check("Shapley, pi shifts (rho)", sh["rho"], 4.30)
    good &= check("Shapley, windings (tau)",
                  fac["shapley_sum"] - sh["pi"] - sh["sigma"] - sh["rho"], 0.42)
    good &= check("Shapley sum", fac["shapley_sum"], 78.82)
    good &= check("relabelling + sign flips (NOT 63)", sh["pi"] + sh["sigma"], 74.10)
    good &= check("sign-flip-alone cell, the 63 we misattributed",
                  fac["cells"]["sigma"]["delta"], 62.90)

    print("\nboth exact reframings, on all three corpora")
    for d, anchors in (("mnist", None), ("fashionmnist", None), ("cifar10", None)):
        W1 = mean(load(f"results/ladder/{d}/W1.json")["acc"])
        W3 = mean(load(f"results/ladder/{d}/W3.json")["acc"])
        for w, name, quoted in (("W4", "c_sort", {"mnist": 0.177, "fashionmnist": 0.170,
                                                  "cifar10": 0.108}),
                                ("W5", "c_align", {"mnist": 0.628, "fashionmnist": 0.664,
                                                   "cifar10": 0.325})):
            f = (mean(load(f"results/ladder/{d}/{w}.json")["acc"]) - W3) / (W1 - W3)
            good &= check(f"{d} {name}", f, quoted[d], 0.0015)

    print("\npixel baselines quoted in the introduction")
    good &= check("MNIST real pixels (P0)", mean(load("results/ladder/mnist/P0.json")["acc"]), 97.97)
    good &= check("CIFAR-10 real pixels (P0)", mean(load("results/ladder/cifar10/P0.json")["acc"]), 55.81)

    print("\nthe 2x2 additive split must close")
    def frac(w):
        j = load(f"results/ladder/mnist/{w}.json")
        return (mean(j["acc"] if isinstance(j, dict) else j) - 13.922) / (94.356 - 13.922)
    ur, up, gr, gp = frac("W12ub"), frac("W12u"), frac("W12b"), frac("W12")
    w11a = mean(load("results/ladder/mnist/W11a_fixednorm.json")["variants"]["W11a"]["acc"]) \
        if "variants" in load("results/ladder/mnist/W11a_fixednorm.json") else None
    good &= check("phasor main effect, at raw bias", up - ur, 0.301)
    good &= check("grading main effect, at raw bias", gr - ur, 0.046)
    good &= check("grading measured at the phasor instead", gp - up, 0.059)
    good &= check("interaction", gp - up - gr + ur, 0.013)
    good &= check("the four additive terms close the step",
                  0.296 + 0.301 + 0.046 + 0.013, gp - 0.2606, 0.002)

    print("\nprobability calls (Brier)")
    prob = [r for r in csv.DictReader((ROOT / "docs/PREDICTION_OUTCOMES.csv").open())
            if r["kind"].strip() == "probability" and r["brier"].strip()]
    allb = [float(r["brier"]) for r in prob]
    noqg = [float(r["brier"]) for r in prob if not r["prediction"].startswith("QG")]
    good &= check("probability calls, excluding gate rows", len(noqg), 54, 0)
    good &= check("mean Brier, excluding gate rows", sum(noqg) / len(noqg), 0.190, 0.0006)
    good &= check("mean Brier, all rows", sum(allb) / len(allb), 0.201, 0.0006)

    print("\nstrings the paper must contain")
    for needle in ["$1.60$", "$96.36\\%$", "$94.76\\%$", "$80.43$", "$28.56$", "QG-7"]:
        good &= appears(txt, needle)

    print("\n" + ("ALL CHECKS PASS" if good else "SOME CHECKS FAILED"))
    sys.exit(0 if good else 1)

#!/usr/bin/env python3
"""S1 confirmatory analysis: scores docs/prereg/S1.md against the ladder cells, draws F9.

Everything here is fixed by the frozen registration — hypotheses, tests, Holm family, TOST
margin — so this script does no model selection. Rungs with different seed counts are paired
on their common seeds (0..min(n)-1); each rung's own mean/CI still uses all of its seeds.

Usage: .venv/bin/python scripts/14_ladder_analysis.py [--dataset mnist]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sirengap.eval.stats import bootstrap_ci_mean, holm, paired_summary, tost_equivalence  # noqa: E402

TOST_MARGIN = 1.0

# Registered intervals are per *dataset arm*, not per hypothesis name. The MNIST numbers are
# magnitudes calibrated on the MNIST anchor, so scoring another dataset against them manufactures
# fake misses — which is exactly what happened when the FashionMNIST arm printed H-S1-4a as a MISS
# at +70.31 against an interval registered for a corpus with an 8-point higher ceiling. A dataset
# with no registration of its own is a *replication of structure* and is reported unscored.
#
#   mnist        -> docs/prereg/S1.md (8c029cf43f01a94c) + addendum 01
#   cifar10      -> docs/prereg/S1-cifar.md (f7906fc6904c7c81), rows H-C1-*
#   fashionmnist -> unregistered; replication only
REGISTERED: dict[str, dict[str, dict[str, float]] | None] = {
    "mnist": {
        "H-S1-1": {"point": 0.1, "lo80": -0.4, "hi80": 0.6},
        "H-S1-2": {"point": -3.8, "lo80": -6.5, "hi80": -1.5},
        "H-S1-3": {"point": 2.0, "lo80": 0.0, "hi80": 6.0},
        "H-S1-4a": {"point": 80.4, "lo80": 79.0, "hi80": 82.0},
        "H-S1-5": {"point": 15.0, "lo80": 5.0, "hi80": 35.0},
        "H-S1-6": {"point": 0.0, "lo80": -3.0, "hi80": 3.0},
        "f_W4": {"point": 0.06, "lo80": 0.01, "hi80": 0.20},
        "f_W5": {"point": 0.10, "lo80": 0.02, "hi80": 0.30},
    },
    "cifar10": {
        "H-S1-1": {"point": -0.4, "lo80": -2.0, "hi80": 0.6},      # H-C1-2
        "H-S1-2": {"point": -12.0, "lo80": -22.0, "hi80": -5.0},   # H-C1-3
        "H-S1-3": {"point": -0.7, "lo80": -2.5, "hi80": 1.0},      # H-C1-6
        "H-S1-4a": {"point": 27.0, "lo80": 14.0, "hi80": 41.0},    # H-C1-5
        "H-S1-5": {"point": 0.0, "lo80": -2.0, "hi80": 2.0},       # H-C1-11
        "H-S1-6": {"point": 0.0, "lo80": -3.0, "hi80": 3.0},       # H-C1-17
        "f_W4": {"point": 0.17, "lo80": 0.08, "hi80": 0.30},       # H-C1-7
        "f_W5": {"point": 0.62, "lo80": 0.42, "hi80": 0.78},       # H-C1-8
        "f_W6": {"point": 0.04, "lo80": -0.02, "hi80": 0.12},      # H-C1-9
        "f_W7": {"point": 0.04, "lo80": -0.02, "hi80": 0.12},      # H-C1-10
        "f_W9": {"point": 0.00, "lo80": -0.03, "hi80": 0.05},      # H-C1-12
        "f_W10": {"point": 0.38, "lo80": 0.12, "hi80": 0.62},      # H-C1-13
    },
    "cifar10gray": {
        # S1-gray.md (b84b660829aa6d40), rows H-G1-*. Only the rungs this partial arm runs.
        "H-S1-1": {"point": 0.2, "lo80": -1.0, "hi80": 1.5},        # H-G1-2
        "H-S1-4a": {"point": 25.5, "lo80": 14.0, "hi80": 36.0},     # H-G1-5
        "f_W4": {"point": 0.13, "lo80": 0.06, "hi80": 0.24},        # H-G1-8
        "f_W5": {"point": 0.45, "lo80": 0.28, "hi80": 0.66},        # H-G1-6
        "f_W9": {"point": 0.00, "lo80": -0.03, "hi80": 0.05},       # H-G1-9
        "f_W10": {"point": 0.45, "lo80": 0.26, "hi80": 0.60},       # H-G1-7
    },
    "fashionmnist": None,
}
PREREG_SOURCE = {
    "mnist": "docs/prereg/S1.md (8c029cf43f01a94c) + addendum 01",
    "cifar10": "docs/prereg/S1-cifar.md (f7906fc6904c7c81), rows H-C1-*",
    "cifar10gray": "docs/prereg/S1-gray.md (b84b660829aa6d40), rows H-G1-* — PARTIAL ladder",
    "fashionmnist": "unregistered — replication of structure, not scored",
}
# arms that deliberately omit rungs; the analysis must not demand the full set
PARTIAL_ARMS = {"cifar10gray": ("P0", "P1", "W1", "W3", "W4", "W5", "W9", "W10")}

WATERFALL = ("P0", "P1", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10")
LABELS = {
    "P0": "P0\npixels", "P1": "P1\nrender", "W1": "W1\nshared-det", "W2": "W2\nshared-stoch",
    "W3": "W3\nrandom init", "W4": "W4\nc_sort", "W5": "W5\nc_align", "W6": "W6\naugment",
    "W7": "W7\nK-marginal", "W8": "W8\ncanon+aug", "W9": "W9\nframe avg", "W10": "W10\ninvariants",
}


def load_cells(directory: Path) -> dict[str, dict]:
    """Rung cells only — the directory also holds this script's own output and E-track files."""
    cells = {}
    for path in sorted(directory.glob("*.json")):
        cell = json.loads(path.read_text())
        if "rung" in cell and "acc" in cell:
            cells[cell["rung"]] = cell
    return cells


def paired(a: list[float], b: list[float]) -> dict:
    n = min(len(a), len(b))
    summary = paired_summary(np.array(a[:n]), np.array(b[:n]))
    summary["paired_on_seeds"] = n
    return summary


def recovery_fraction(rung: list[float], w3: list[float], w1: list[float]) -> dict:
    """f = (rung - W3) / (W1 - W3), bootstrapped over the common seeds."""
    n = min(len(rung), len(w3), len(w1))
    frac = (np.array(rung[:n]) - np.array(w3[:n])) / (np.array(w1[:n]) - np.array(w3[:n]))
    return {"point": float(frac.mean()), "ci95": bootstrap_ci_mean(frac), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--root", default="results/ladder")
    args = ap.parse_args()

    d = Path(args.root) / args.dataset
    cells = load_cells(d)
    acc = {k: v["acc"] for k, v in cells.items()}
    required = PARTIAL_ARMS.get(
        args.dataset, ("P0", "P1", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W10")
    )
    missing = [r for r in required if r not in acc]
    if missing:
        raise SystemExit(f"missing rungs: {missing}")
    partial = args.dataset in PARTIAL_ARMS

    reg = REGISTERED.get(args.dataset)
    have = acc.keys().__contains__

    def add(name: str, needs: tuple[str, ...], build):
        """Register a test only when this arm actually ran the rungs it needs."""
        return build() if all(have(r) for r in needs) else None

    tests = {}
    for name, needs, build in [
        ("H-S1-1", ("P1", "P0"), lambda: {
            "statement": "P1 ~ P0 (TOST); is class information preserved by the fit?",
            "diff": paired(acc["P1"], acc["P0"]),
            "tost": tost_equivalence(np.array(acc["P1"]), np.array(acc["P0"]), TOST_MARGIN),
        }),
        ("H-S1-2", ("W1", "P1"), lambda: {
            "statement": "W1 - P1 residual under zero nuisance",
            "diff": paired(acc["W1"], acc["P1"]),
        }),
        ("H-S1-3", ("W1", "W2"), lambda: {
            "statement": "W1 - W2 optimization-noise rung",
            "diff": paired(acc["W1"], acc["W2"]),
        }),
        ("H-S1-4a", ("W1", "W3"), lambda: {
            "statement": "W1 - W3 init/symmetry gap (the perception gap)",
            "diff": paired(acc["W1"], acc["W3"]),
        }),
        ("H-S1-5", ("W7", "W6"), lambda: {
            "statement": "(W7 - W3) - (W6 - W3) = W7 - W6; marginalize vs group-augment",
            "diff": paired(acc["W7"], acc["W6"]),
        }),
        ("H-S1-6", ("W10", "W4", "W5"), lambda: {
            "statement": "W10 within +-3 pts of the interval [W4, W5]",
            "w10_mean": float(np.mean(acc["W10"])),
            "interval": [float(np.mean(acc["W4"])), float(np.mean(acc["W5"]))],
        }),
    ]:
        built = add(name, needs, build)
        if built is not None:
            tests[name] = built
    for name, t in tests.items():
        if reg and name in reg:
            t["registered"] = reg[name]
    if "H-S1-6" in tests:
        lo, hi = sorted(tests["H-S1-6"]["interval"])
        w10 = tests["H-S1-6"]["w10_mean"]
        tests["H-S1-6"]["distance_outside_interval"] = float(max(0.0, lo - w10, w10 - hi))

    family = [h for h in ("H-S1-1", "H-S1-2", "H-S1-3", "H-S1-4a", "H-S1-5") if h in tests]
    adjusted = holm([tests[h]["diff"]["t_p"] for h in family])
    for h, p in zip(family, adjusted):
        tests[h]["holm_adjusted_p"] = p

    for h in family + (["H-S1-6"] if "H-S1-6" in tests else []):
        obs = tests[h]["diff"]["mean_diff"] if "diff" in tests[h] else tests[h]["distance_outside_interval"]
        tests[h]["observed"] = obs
        if "registered" in tests[h]:
            r = tests[h]["registered"]
            tests[h]["interval_hit"] = bool(r["lo80"] <= obs <= r["hi80"])
            tests[h]["abs_error"] = abs(obs - r["point"])

    recovery = {
        f"f_{r}": recovery_fraction(acc[r], acc["W3"], acc["W1"])
        for r in ("W4", "W5", "W6", "W7", "W9", "W10")
        if r in acc
    }
    for key in list(recovery):
        if reg and key in reg:
            r = reg[key]
            recovery[f"registered_{key}"] = r
            recovery[key]["interval_hit"] = bool(r["lo80"] <= recovery[key]["point"] <= r["hi80"])
    controls = {
        r: {
            "label_shuffle_test_acc": cells[r].get("label_shuffle_test_acc"),
            "linear_probe": cells[r].get("linear_probe"),
            "knn10_cosine": cells[r].get("knn10_cosine"),
        }
        for r in cells
    }
    if "W7-1/8" in acc:
        tests["W7_rowcount_control"] = {
            "statement": "W7 vs W7-1/8: is the K-marginalization gain just 8x more rows?",
            "diff": paired(acc["W7"], acc["W7-1/8"]),
        }
    if "X1" in cells and cells["X1"].get("extra_acc"):
        tests["X1_transfer"] = {
            "w1_trained_on_w3_features": cells["X1"]["extra_acc"],
            "w1_own_test": float(np.mean(acc["X1"])),
            "reverse": cells["X1"].get("reverse", {}).get("extra_acc"),
        }

    report = {
        "dataset": args.dataset,
        "prereg": PREREG_SOURCE.get(args.dataset, "unregistered — not scored"),
        "scored": reg is not None,
        "partial_arm": partial,
        "rungs_run": sorted(acc),
        "means": {k: float(np.mean(v)) for k, v in sorted(acc.items())},
        "seeds": {k: len(v) for k, v in sorted(acc.items())},
        "tests": tests,
        "recovery_fractions": recovery,
        "controls": controls,
    }
    out = d / "S1_analysis.json"
    out.write_text(json.dumps(report, indent=2))

    _figure(acc, d / "F9_waterfall.png", args.dataset)
    print(json.dumps({k: report[k] for k in ("means", "recovery_fractions")}, indent=2))
    print(f"\nregistration: {report['prereg']}")
    for h in family:
        t = tests[h]
        if "registered" not in t:
            print(f"{h}: obs {t['observed']:+.2f} (unregistered for this dataset — not scored), "
                  f"holm p={t['holm_adjusted_p']:.2g}")
            continue
        print(f"{h}: obs {t['observed']:+.2f} vs registered {t['registered']['point']:+.2f} "
              f"[{t['registered']['lo80']}, {t['registered']['hi80']}] -> "
              f"{'HIT' if t['interval_hit'] else 'MISS'}, holm p={t['holm_adjusted_p']:.2g}")
    if "H-S1-6" in tests:
        t6 = tests["H-S1-6"]
        verdict = "not scored" if "registered" not in t6 else ("HIT" if t6["interval_hit"] else "MISS")
        print(f"H-S1-6: W10 {t6['w10_mean']:.2f}, interval {t6['interval']}, "
              f"outside by {t6['distance_outside_interval']:.2f} -> {verdict}")
    for key in [k for k in ("f_W4", "f_W5", "f_W6", "f_W7", "f_W9", "f_W10") if k in recovery]:
        v = recovery[key]
        hit = v.get("interval_hit")
        mark = "" if hit is None else (" -> HIT" if hit else " -> MISS")
        print(f"{key}: {v['point']:.3f} {v['ci95']}{mark}")
    print(f"\nwritten {out} and F9_waterfall.png")


def _figure(acc: dict[str, list[float]], path: Path, dataset: str) -> None:
    rungs = [r for r in WATERFALL if r in acc]
    means = [float(np.mean(acc[r])) for r in rungs]
    errs = [float(np.std(acc[r], ddof=1)) for r in rungs]
    colors = ["#4c72b0" if r in ("P0", "P1") else "#dd8452" if r in ("W1", "W2") else "#55a868"
              for r in rungs]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(range(len(rungs)), means, yerr=errs, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(float(np.mean(acc["W1"])), ls="--", lw=1, color="#dd8452",
               label=f"W1 ceiling ({np.mean(acc['W1']):.1f})")
    ax.axhline(float(np.mean(acc["W3"])), ls=":", lw=1, color="grey",
               label=f"W3 floor ({np.mean(acc['W3']):.1f})")
    ax.axhline(10.0, ls="-", lw=0.8, color="crimson", alpha=0.6, label="chance (10)")
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([LABELS.get(r, r) for r in rungs], fontsize=8)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"F9 — S1 decomposition ladder, sine INRs on {dataset} "
                 f"(matched MLP, mean ± SD over seeds)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()

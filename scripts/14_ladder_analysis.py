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
WATERFALL = ("P0", "P1", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10")
LABELS = {
    "P0": "P0\npixels", "P1": "P1\nrender", "W1": "W1\nshared-det", "W2": "W2\nshared-stoch",
    "W3": "W3\nrandom init", "W4": "W4\nc_sort", "W5": "W5\nc_align", "W6": "W6\naugment",
    "W7": "W7\nK-marginal", "W8": "W8\ncanon+aug", "W9": "W9\nframe avg", "W10": "W10\ninvariants",
}


def load_cells(directory: Path) -> dict[str, dict]:
    cells = {}
    for path in directory.glob("*.json"):
        cell = json.loads(path.read_text())
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
    missing = [r for r in ("P0", "P1", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W10") if r not in acc]
    if missing:
        raise SystemExit(f"missing rungs: {missing}")

    tests = {
        "H-S1-1": {
            "statement": "P1 ~ P0 (TOST margin 1.0 pt); registered +0.1 [-0.4, +0.6]",
            "diff": paired(acc["P1"], acc["P0"]),
            "tost": tost_equivalence(np.array(acc["P1"]), np.array(acc["P0"]), TOST_MARGIN),
            "registered": {"point": 0.1, "lo80": -0.4, "hi80": 0.6},
        },
        "H-S1-2": {
            "statement": "W1 - P1 residual; registered -3.8 [-6.5, -1.5]",
            "diff": paired(acc["W1"], acc["P1"]),
            "registered": {"point": -3.8, "lo80": -6.5, "hi80": -1.5},
        },
        "H-S1-3": {
            "statement": "W1 - W2 optimization-noise rung; registered +2.0 [0.0, +6.0]",
            "diff": paired(acc["W1"], acc["W2"]),
            "registered": {"point": 2.0, "lo80": 0.0, "hi80": 6.0},
        },
        "H-S1-4a": {
            "statement": "W1 - W3 init/symmetry gap; registered +80.4 [79, 82]",
            "diff": paired(acc["W1"], acc["W3"]),
            "registered": {"point": 80.4, "lo80": 79.0, "hi80": 82.0},
        },
        "H-S1-5": {
            "statement": "(W7 - W3) - (W6 - W3) = W7 - W6; registered +15 [+5, +35]",
            "diff": paired(acc["W7"], acc["W6"]),
            "registered": {"point": 15.0, "lo80": 5.0, "hi80": 35.0},
        },
        "H-S1-6": {
            "statement": "W10 within +-3 pts of the interval [W4, W5]",
            "w10_mean": float(np.mean(acc["W10"])),
            "interval": [float(np.mean(acc["W4"])), float(np.mean(acc["W5"]))],
            "registered": {"point": 0.0, "lo80": -3.0, "hi80": 3.0},
        },
    }
    lo, hi = sorted(tests["H-S1-6"]["interval"])
    w10 = tests["H-S1-6"]["w10_mean"]
    tests["H-S1-6"]["distance_outside_interval"] = float(max(0.0, lo - w10, w10 - hi))

    family = ["H-S1-1", "H-S1-2", "H-S1-3", "H-S1-4a", "H-S1-5"]
    adjusted = holm([tests[h]["diff"]["t_p"] for h in family])
    for h, p in zip(family, adjusted):
        tests[h]["holm_adjusted_p"] = p

    for h in family + ["H-S1-6"]:
        reg = tests[h]["registered"]
        obs = tests[h]["diff"]["mean_diff"] if "diff" in tests[h] else tests[h]["distance_outside_interval"]
        tests[h]["observed"] = obs
        tests[h]["interval_hit"] = bool(reg["lo80"] <= obs <= reg["hi80"])
        tests[h]["abs_error"] = abs(obs - reg["point"])

    recovery = {
        "f_W4": recovery_fraction(acc["W4"], acc["W3"], acc["W1"]),
        "f_W5": recovery_fraction(acc["W5"], acc["W3"], acc["W1"]),
        "f_W6": recovery_fraction(acc["W6"], acc["W3"], acc["W1"]),
        "f_W7": recovery_fraction(acc["W7"], acc["W3"], acc["W1"]),
        "f_W9": recovery_fraction(acc["W9"], acc["W3"], acc["W1"]),
        "f_W10": recovery_fraction(acc["W10"], acc["W3"], acc["W1"]),
        "registered_f_W4": {"point": 0.06, "lo80": 0.01, "hi80": 0.20},
        "registered_f_W5": {"point": 0.10, "lo80": 0.02, "hi80": 0.30},
    }
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
        "prereg": "docs/prereg/S1.md (8c029cf43f01a94c) + addendum 01",
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
    for h in family:
        t = tests[h]
        print(f"{h}: obs {t['observed']:+.2f} vs registered {t['registered']['point']:+.2f} "
              f"[{t['registered']['lo80']}, {t['registered']['hi80']}] -> "
              f"{'HIT' if t['interval_hit'] else 'MISS'}, holm p={t['holm_adjusted_p']:.2g}")
    t6 = tests["H-S1-6"]
    print(f"H-S1-6: W10 {t6['w10_mean']:.2f}, interval {t6['interval']}, "
          f"outside by {t6['distance_outside_interval']:.2f} -> "
          f"{'HIT' if t6['interval_hit'] else 'MISS'}")
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

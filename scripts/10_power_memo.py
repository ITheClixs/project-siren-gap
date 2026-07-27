#!/usr/bin/env python3
"""S1 power memo (protocol §0.5): seed-count sizing from the measured anchor variance.

Reads the A1/A2 anchor seed vectors, estimates the paired-difference SD (with an upper
confidence limit, because an SD from 5 seeds is itself noisy), and reports:

  * MDE  — smallest true difference detectable at alpha=.05 two-sided with power .8,
           for a paired t-test at n seeds (noncentral-t, exact);
  * TOST power — probability of *declaring equivalence* at margin m when the true
           difference is delta, for a paired TOST at n seeds (simulated).

Usage: .venv/bin/python scripts/10_power_memo.py [--anchors results/anchors/anchors_mnist.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy import stats as sps

ALPHA = 0.05
TARGET_POWER = 0.80
SEED_GRID = (5, 8, 10, 15, 20)
SIGMA_LABELS = ("A1 (W1-W3)", "A2 (W6-W3)")


def sd_with_upper_limit(diffs: np.ndarray, conf: float = 0.80) -> tuple[float, float]:
    """Sample SD and its upper `conf` confidence limit (chi-square, one-sided)."""
    n = len(diffs)
    sd = float(diffs.std(ddof=1))
    chi2_lo = sps.chi2.ppf(1.0 - conf, n - 1)
    return sd, float(sd * np.sqrt((n - 1) / chi2_lo))


# scipy's noncentral t returns NaN for ncp >~ 9 at these df; power there is 1 to >7 decimals
# (checked: ncp=8 gives 0.99988 at df=4), so saturate rather than propagate NaN.
NCP_SATURATION = 8.0


def paired_t_power(effect_pts: float, sd: float, n: int, alpha: float = ALPHA) -> float:
    """Exact power of a two-sided paired t-test via the noncentral t distribution."""
    ncp = effect_pts / sd * np.sqrt(n)
    if ncp > NCP_SATURATION:
        return 1.0
    crit = sps.t.ppf(1.0 - alpha / 2.0, n - 1)
    return float(sps.nct.sf(crit, n - 1, ncp) + sps.nct.cdf(-crit, n - 1, ncp))


def mde(sd: float, n: int, power: float = TARGET_POWER) -> float:
    """Smallest detectable difference in accuracy points at the target power."""
    f = lambda e: paired_t_power(e, sd, n) - power  # noqa: E731
    hi = sd * NCP_SATURATION / np.sqrt(n)
    return float(optimize.brentq(f, 1e-9, hi))


def tost_power(delta: float, sd: float, n: int, margin: float, n_sim: int = 40000, seed: int = 0) -> float:
    """P(declare equivalence) for a paired TOST at margin `margin`, true difference `delta`."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(delta, sd, size=(n_sim, n))
    mean = draws.mean(axis=1)
    se = draws.std(axis=1, ddof=1) / np.sqrt(n)
    crit = sps.t.ppf(1.0 - ALPHA, n - 1)
    return float((((mean + margin) / se > crit) & ((mean - margin) / se < -crit)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="results/anchors/anchors_mnist.json")
    ap.add_argument("--margin", type=float, default=1.0, help="TOST equivalence margin (pts)")
    ap.add_argument("--out", default="results/power/S1_power.json")
    args = ap.parse_args()

    anchors = json.loads(Path(args.anchors).read_text())
    acc = {  # W3 is shared: the A2 run reuses the A1 W3 seed vector
        "W1": np.array(anchors["A1"]["acc_W1"]),
        "W3": np.array(anchors["A1"]["acc_W3"]),
        "W6": np.array(anchors["A2"]["acc_W6"]),
    }
    diff_sets = {
        SIGMA_LABELS[0]: acc["W1"] - acc["W3"],
        SIGMA_LABELS[1]: acc["W6"] - acc["W3"],
    }
    per_rung_sd = {rung: float(v.std(ddof=1)) for rung, v in acc.items()}

    sigmas: dict[str, dict] = {}
    for label, diffs in diff_sets.items():
        sd, sd_hi = sd_with_upper_limit(diffs)
        sigmas[label] = {"n": len(diffs), "sd": sd, "sd_upper80": sd_hi}

    # Two noise classes: rungs whose decoder input is a fixed feature matrix (A1 anchor
    # measures their seed noise) vs rungs that resample group elements every step
    # (A2 anchor). Sizing them together would over-buy seeds for the deterministic half.
    classes = {
        "deterministic": sigmas[SIGMA_LABELS[0]],
        "augmentation_bearing": sigmas[SIGMA_LABELS[1]],
    }

    def size_class(planning_sd: float, observed_sd: float) -> dict:
        table = [
            {
                "n_seeds": n,
                "mde_at_planning_sd": mde(planning_sd, n),
                "mde_at_observed_sd": mde(observed_sd, n),
                "tost_power_delta0": tost_power(0.0, planning_sd, n, args.margin),
                "tost_power_delta_quarter_margin": tost_power(
                    0.25 * args.margin, planning_sd, n, args.margin
                ),
            }
            for n in SEED_GRID
        ]
        adequate = [
            row["n_seeds"]
            for row in table
            if row["mde_at_planning_sd"] <= args.margin and row["tost_power_delta0"] >= TARGET_POWER
        ]
        return {"table": table, "smallest_adequate_n": adequate[0] if adequate else None}

    sizing = {
        name: {
            "planning_sd_pts": v["sd_upper80"],
            "observed_sd_pts": v["sd"],
            **size_class(v["sd_upper80"], v["sd"]),
        }
        for name, v in classes.items()
    }

    report = {
        "anchors_file": args.anchors,
        "per_rung_seed_sd": per_rung_sd,
        "paired_diff_sd": sigmas,
        "tost_margin_pts": args.margin,
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "sizing": sizing,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

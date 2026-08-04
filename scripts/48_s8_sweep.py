#!/usr/bin/env python3
"""S8: decode the ladder at every step budget, and score it (docs/prereg/S8.md).

For each budget the corpora differ from the main ladder's only in the number of optimizer
steps and in size (10k/2k/2k, so the 10000-step arm is affordable). Comparisons are *within*
the sweep: the 300-step arm at this reduced size is the internal control, never the
full-corpus ladder.

Reported per budget: W1, W3, W4, W5, W10 with the frozen apparatus, plus three fit
diagnostics that need no new fitting -- median relative endpoint gradient norm (the
stationarity measure), median render PSNR, and median relative parameter travel from theta_0.

Usage:
  .venv/bin/python scripts/48_s8_sweep.py --budgets 300 1000 3000 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sirengap.canon.calign import c_align  # noqa: E402
from sirengap.canon.csort import c_sort  # noqa: E402
from sirengap.canon.deep_invariants import encode_deep  # noqa: E402
from sirengap.eval.decoder import train_matched_mlp  # noqa: E402
from sirengap.eval.rungs import (  # noqa: E402
    CorpusCache,
    _chunked,
    probe_coords,
    shared_init_template,
)
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402

SEEDS = 5
RUNGS = ("W1", "W3", "W4", "W5", "W10")

# frozen in docs/prereg/S8.md before any cell was decoded
REGISTERED = {
    "H-S8-1": ("W1 accuracy at 300 steps", 88.0, (82.0, 93.0)),
    "H-S8-2": ("W1 - W3 gap at 10000 steps", 75.0, (60.0, 85.0)),
    "H-S8-3": ("f(W5) at 300 steps", 0.60, (0.45, 0.72)),
    "H-S8-4": ("f(W5) at 10000 steps", 0.45, (0.20, 0.65)),
    "H-S8-5": ("f(W5) at 10000 minus at 300", -0.15, (-0.40, 0.05)),
    "H-S8-6": ("median relative endpoint gradient norm at 10000", 3e-4, (3e-5, 3e-3)),
    "H-S8-7": ("median relative parameter travel at 10000", 0.35, (0.20, 0.60)),
    "H-S8-8": ("f(W10) at 10000 steps", 0.25, (0.05, 0.45)),
}
FALSIFIER = 0.15  # f(W5) below this at the largest budget rescopes every ladder claim
# registered in docs/prereg/S8.md section 5, before any corpus was fitted
REGISTERED_P = {
    "P-S8-A": ("probability f(W5) at 10000 steps is still above 0.30", 0.70),
    "P-S8-B": ("probability the W1-W3 gap at 10000 steps is still at least 50 points", 0.80),
    "P-S8-C": ("probability the median relative gradient norm falls 10x from 300 to 10000 steps",
               0.60),
}
OUTCOMES = ROOT / "docs" / "PREDICTION_OUTCOMES.csv"
DATE = "2026-08-04"


def ledger_rows(report: dict) -> list[dict]:
    """The frozen S8 calls as PREDICTION_OUTCOMES rows. Scored once, when all arms exist."""
    rows = []
    for key, r in report.get("registered", {}).items():
        rows.append({
            "date_scored": DATE, "prediction": f"{key} {r['statement']}", "kind": "interval",
            "point": r["point"], "lo80": r["interval"][0], "hi80": r["interval"][1],
            "observed": round(float(r["observed"]), 6), "verdict": r["verdict"],
            "abs_error": round(abs(float(r["observed"]) - r["point"]), 6), "brier": "", "note": "",
        })
    for key, (stmt, p) in REGISTERED_P.items():
        if key not in report:
            continue
        happened = bool(report[key])
        rows.append({
            "date_scored": DATE, "prediction": f"{key} {stmt}", "kind": "probability",
            "point": p, "lo80": "", "hi80": "", "observed": int(happened), "verdict": "—",
            "abs_error": "", "brier": round((p - happened) ** 2, 4),
            "note": report.get("ledger_note", {}).get(key, ""),
        })
    return rows


def append_ledger(report: dict, rescore: bool) -> None:
    """Write the S8 rows, with the append-once guard of CLAIMS row 54."""
    import csv

    rows = ledger_rows(report)
    if not rows:
        return
    with OUTCOMES.open(newline="") as fh:
        already = {r["prediction"] for r in csv.DictReader(fh)}
    clash = sorted({r["prediction"] for r in rows} & already)
    if clash and not rescore:
        raise SystemExit(
            f"refusing to double-score: {len(clash)} S8 predictions are already in "
            f"{OUTCOMES.name} (first: {clash[0]!r}). Pass --rescore only after removing the "
            "superseded rows."
        )
    with OUTCOMES.open(newline="") as fh:
        fieldnames = list(csv.DictReader(fh).fieldnames)
    with OUTCOMES.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writerows(rows)
    print(f"appended {len(rows)} rows to {OUTCOMES}")


def diagnostics(cache: CorpusCache, protocol: str, root: Path, dataset: str) -> dict:
    """Stationarity, fidelity and travel -- all from the corpus, no new fitting."""
    params, meta = cache.corpus(protocol)
    theta0 = shared_init_template(params)
    flat = params.flat().double()
    rel_travel = (flat - theta0.flat().double()).norm(dim=1) / theta0.flat().double().norm()
    out = {
        "psnr_median": float(meta["final_psnr"].median()),
        "travel_median": float(rel_travel.median()),
        "travel_q95": float(torch.quantile(rel_travel, 0.95)),
    }
    if "final_grad_norm" in meta.columns:
        out["grad_norm_median"] = float(meta["final_grad_norm"].median())
        out["grad_norm_q95"] = float(meta["final_grad_norm"].quantile(0.95))
    return out


def decode(feats: dict, labels: dict, device: str, seeds: int) -> dict:
    accs = [train_matched_mlp(feats, labels, seed=s, device=device).test_acc for s in range(seeds)]
    a = np.array(accs)
    return {"acc": accs, "mean": float(a.mean()), "ci95": bootstrap_ci_mean(a)}


def run_budget(steps: int, dataset: str, root: Path, device: str, seeds: int,
               smoke: bool = False) -> dict:
    cache = CorpusCache(root / dataset, dataset)
    shared, random_ = f"P-shared-det-s8s{steps}", f"P-random-s8s{steps}"
    sh_split, sh_labels = cache.split_params(shared)
    rn_split, rn_labels = cache.split_params(random_)

    if smoke:  # nothing decoded here estimates a registered quantity
        g = torch.Generator().manual_seed(0)
        for lab in (sh_labels, rn_labels):
            lab["train"] = lab["train"][torch.randperm(len(lab["train"]), generator=g)]

    template = shared_init_template(rn_split["train"])
    probes = probe_coords(dataset)

    t0 = time.time()
    cells: dict[str, dict] = {}
    cells["W1"] = decode({s: p.flat() for s, p in sh_split.items()}, sh_labels, device, seeds)
    cells["W3"] = decode({s: p.flat() for s, p in rn_split.items()}, rn_labels, device, seeds)
    cells["W4"] = decode(
        {s: _chunked(lambda q: c_sort(q)[0].flat(), p) for s, p in rn_split.items()},
        rn_labels, device, seeds)
    cells["W5"] = decode(
        {s: _chunked(lambda q: c_align(q, template, probes)[0].flat(), p)
         for s, p in rn_split.items()},
        rn_labels, device, seeds)
    cells["W10"] = decode(
        {s: _chunked(encode_deep, p) for s, p in rn_split.items()}, rn_labels, device, seeds)

    w1, w3 = np.array(cells["W1"]["acc"]), np.array(cells["W3"]["acc"])
    frac = {}
    for r in ("W4", "W5", "W10"):
        a = np.array(cells[r]["acc"])
        n = min(len(a), len(w1), len(w3))
        frac[r] = float(((a[:n] - w3[:n]) / (w1[:n] - w3[:n])).mean())

    return {
        "steps": steps,
        "cells": cells,
        "gap": float(w1.mean() - w3.mean()),
        "f": frac,
        "diagnostics": {"shared": diagnostics(cache, shared, root, dataset),
                        "random": diagnostics(cache, random_, root, dataset)},
        "wallclock_s": time.time() - t0,
    }


def verdict(v: float, lo: float, hi: float) -> str:
    return "HIT" if lo <= v <= hi else "MISS"


def score(by_steps: dict[int, dict]) -> dict:
    obs: dict[str, float] = {}
    if 300 in by_steps:
        obs["H-S8-1"] = by_steps[300]["cells"]["W1"]["mean"]
        obs["H-S8-3"] = by_steps[300]["f"]["W5"]
    if 10000 in by_steps:
        top = by_steps[10000]
        obs["H-S8-2"] = top["gap"]
        obs["H-S8-4"] = top["f"]["W5"]
        obs["H-S8-8"] = top["f"]["W10"]
        if "grad_norm_median" in top["diagnostics"]["random"]:
            obs["H-S8-6"] = top["diagnostics"]["random"]["grad_norm_median"]
        obs["H-S8-7"] = top["diagnostics"]["shared"]["travel_median"]
        if 300 in by_steps:
            obs["H-S8-5"] = top["f"]["W5"] - by_steps[300]["f"]["W5"]

    out = {}
    for k, v in obs.items():
        stmt, point, (lo, hi) = REGISTERED[k]
        out[k] = {"statement": stmt, "point": point, "interval": [lo, hi],
                  "observed": v, "verdict": verdict(v, lo, hi)}
        print(f"  {k}: registered {point} [{lo}, {hi}], observed {v:.4g} -> {out[k]['verdict']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--budgets", nargs="+", type=int, default=[300, 1000, 3000, 10000])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--smoke", action="store_true",
                    help="engineering check: shuffles the training labels, so nothing decoded is "
                         "an estimate of a registered quantity, and writes no report")
    ap.add_argument("--no-ledger", action="store_true",
                    help="write sweep.json but not the PREDICTION_OUTCOMES rows")
    ap.add_argument("--rescore", action="store_true",
                    help="permit appending predictions already in the ledger; the caller must "
                         "remove the superseded rows first")
    args = ap.parse_args()

    # The mechanical guard of docs/prereg/S8-addendum-01.md. A pre-run check on registered
    # quantities is exposure even when it is not meant as a measurement; twice in one day a
    # written lesson failed to prevent it, so the guard lives in code now.
    downgraded = args.device == "cpu" and torch.backends.mps.is_available()
    if not args.smoke and (args.seeds != SEEDS or downgraded):
        raise SystemExit(
            f"refusing to write a report off-protocol (seeds={args.seeds}, device={args.device}); "
            f"the registration fixes n={SEEDS}. Use --smoke for an engineering check, which "
            "shuffles the training labels and writes nothing. (A CPU or CUDA replication on a "
            "machine without MPS is not off-protocol and is allowed.)"
        )

    root = Path(args.root)
    out_dir = ROOT / "results" / "s8"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_steps: dict[int, dict] = {}
    for steps in args.budgets:
        if not (root / args.dataset / f"P-random-s8s{steps}").exists():
            print(f"skipping {steps}: corpus not present")
            continue
        r = run_budget(steps, args.dataset, root, args.device, args.seeds, args.smoke)
        by_steps[steps] = r
        d = r["diagnostics"]["random"]
        print(f"steps={steps:6d}  W1={r['cells']['W1']['mean']:.2f}  "
              f"W3={r['cells']['W3']['mean']:.2f}  gap={r['gap']:.2f}  "
              f"f(W4)={r['f']['W4']:.3f}  f(W5)={r['f']['W5']:.3f}  f(W10)={r['f']['W10']:.3f}  "
              f"|grad|={d.get('grad_norm_median', float('nan')):.2e}  "
              f"psnr={d['psnr_median']:.1f}dB  "
              f"travel={r['diagnostics']['shared']['travel_median']:.3f}", flush=True)

    report = {"study": "S8", "prereg": "docs/prereg/S8.md", "dataset": args.dataset,
              "by_steps": {str(k): v for k, v in by_steps.items()}}
    if {300, 10000} & set(by_steps):
        print("\nscoring:")
        report["registered"] = score(by_steps)
        hits = sum(1 for v in report["registered"].values() if v["verdict"] == "HIT")
        report["score"] = f"{hits}/{len(report['registered'])}"
        print(f"S8 intervals: {report['score']}")
    if 10000 in by_steps and 300 in by_steps:
        top, base = by_steps[10000], by_steps[300]
        report["falsifier_fired"] = bool(top["f"]["W5"] < FALSIFIER)
        report["P-S8-A"] = bool(top["f"]["W5"] > 0.30)
        report["P-S8-B"] = bool(top["gap"] >= 50.0)
        gn_t = top["diagnostics"]["random"].get("grad_norm_median")
        gn_b = base["diagnostics"]["random"].get("grad_norm_median")
        if gn_t and gn_b:
            report["P-S8-C"] = bool(gn_b / gn_t >= 10.0)
            report["grad_norm_ratio_300_over_10000"] = gn_b / gn_t
        print(f"falsifier f(W5)<{FALSIFIER}: "
              f"{'FIRED' if report['falsifier_fired'] else 'not fired'}; "
              f"P-S8-A {report['P-S8-A']}, P-S8-B {report['P-S8-B']}, "
              f"P-S8-C {report.get('P-S8-C')}")

    if args.smoke:
        print("\nsmoke run: labels were shuffled and no report is written")
        return
    (out_dir / "sweep.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_dir / 'sweep.json'}")

    # The ledger is written only when every registered arm exists, so a partial sweep cannot
    # score a call that the missing budget would have decided. Transcribing these by hand is
    # what went wrong on the S5 path (CLAIMS row 54), so the scorer does it.
    if {300, 10000} <= set(by_steps) and not args.no_ledger:
        append_ledger(report, args.rescore)
    elif not args.no_ledger:
        missing = {300, 10000} - set(by_steps)
        print(f"ledger not written: the registered set needs budgets {sorted(missing)}")


if __name__ == "__main__":
    main()

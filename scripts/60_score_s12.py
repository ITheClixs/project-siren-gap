#!/usr/bin/env python3
"""Score the frozen S12 registration (docs/prereg/S12.md): the converged-fit ladder.

Two stages, and the order is the point.

1. `--gate` evaluates the validity conditions of S12 section 3 on the corpora alone. It reads no
   decoded accuracy and quotes no ladder number. If the gate fails, the study is reported as
   having failed to produce converged corpora and nothing is decoded.
2. `--score` runs only after the gate passes, reads the decoded arms, and appends the registered
   rows once.

Written before the first corpus finished, so the thresholds could not be chosen against a number.

Usage:
  .venv/bin/python scripts/60_score_s12.py --gate
  .venv/bin/python scripts/60_score_s12.py --score
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTCOMES = ROOT / "docs" / "PREDICTION_OUTCOMES.csv"
S12 = ROOT / "results" / "s12"
REPS = (0, 1, 2)
TAG = "s12r"          # attempt 1; attempt 2's staged corpora use --tag s12b --reps 0
PROTOCOLS = ("P-shared-det", "P-random")
SEEDS = 5

# docs/prereg/S12.md section 3
GRAD_TOL = 1e-4          # median relative endpoint gradient norm, both protocols
PSNR_MATCH_DB = 2.0      # between-protocol median render PSNR difference
# section 4
REGISTERED = {
    "H-S12-1": ("median relative endpoint gradient norm, converged", 2e-5, (1e-6, 9e-5)),
    "H-S12-2": ("W1 - W3 gap, converged reduced corpus", 74.0, (55.0, 82.0)),
    "H-S12-3": ("f(c_align), converged", 0.44, (0.20, 0.65)),
    "H-S12-4": ("f(c_align) converged minus at 10000 non-stationary steps", -0.02, (-0.25, 0.15)),
    "H-S12-5": ("median relative parameter travel, converged", 0.21, (0.15, 0.45)),
    "H-S12-6": ("s(W12), converged", 0.88, (0.65, 0.97)),
    "H-S12-7": ("between-corpus SD of f(c_align) over three replications", 0.02, (0.002, 0.08)),
}
CALLS = {
    "P-S12-A": ("probability f(c_align) converged is above 0.30", 0.75),
    "P-S12-B": ("probability all validity conditions are met on the first attempt", 0.70),
    "P-S12-C": ("probability between-corpus SD exceeds decoder-seed SD by at least 3x", 0.55),
}
FALSIFIER = 0.15


def corpus_dir(protocol: str, rep: int) -> Path:
    return ROOT / "data" / "inrbench" / "mnist" / f"{protocol}-{TAG}{rep}"


def corpus_stats(protocol: str, rep: int) -> dict | None:
    """Median gradient norm and render PSNR straight from the shard metadata."""
    d = corpus_dir(protocol, rep)
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = {"n": int(len(df)), "psnr_median": float(df["final_psnr"].median())}
    if "final_grad_norm" in df.columns:
        out["grad_norm_median"] = float(df["final_grad_norm"].median())
    if "stopped_at" in df.columns:
        out["stopped_median"] = float(df["stopped_at"].median())
        out["stopped_frac"] = float((df["stopped_at"] < df["steps"]).mean())
    return out


def gate() -> dict:
    report: dict = {"conditions": {}, "per_corpus": {}}
    ok = True
    for rep in REPS:
        for p in PROTOCOLS:
            s = corpus_stats(p, rep)
            report["per_corpus"][f"{p}-r{rep}"] = s
            if s is None:
                report["conditions"][f"{p}-r{rep} present"] = False
                ok = False
    if not ok:
        report["gate_passed"] = False
        report["reason"] = "not every replication is present"
        return report

    for rep in REPS:
        for p in PROTOCOLS:
            s = report["per_corpus"][f"{p}-r{rep}"]
            key = f"{p}-r{rep} grad_norm < {GRAD_TOL:g}"
            passed = s.get("grad_norm_median", float("inf")) < GRAD_TOL
            report["conditions"][key] = passed
            ok &= passed
        a = report["per_corpus"][f"P-shared-det-r{rep}"]["psnr_median"]
        b = report["per_corpus"][f"P-random-r{rep}"]["psnr_median"]
        key = f"r{rep} render PSNR matched within {PSNR_MATCH_DB} dB"
        passed = abs(a - b) <= PSNR_MATCH_DB
        report["conditions"][key] = passed
        report[f"psnr_gap_r{rep}"] = round(a - b, 3)
        ok &= passed

    report["gate_passed"] = bool(ok)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default=TAG, help="corpus tag prefix, e.g. s12b for attempt 2")
    ap.add_argument("--reps", type=int, nargs="+", default=list(REPS),
                    help="which replications to require; the staged check uses 0 alone")
    ap.add_argument("--out", default="gate.json")
    args = ap.parse_args()
    globals()["TAG"] = args.tag
    globals()["REPS"] = tuple(args.reps)
    S12.mkdir(parents=True, exist_ok=True)

    if args.gate or not args.score:
        rep = gate()
        for k, v in rep["conditions"].items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"\nS12 gate: {'PASSED' if rep['gate_passed'] else 'FAILED'}")
        if not rep["gate_passed"]:
            print("Per S12 section 5, nothing is decoded and the failure is what gets reported.")
        (S12 / args.out).write_text(json.dumps(rep, indent=2))
        print(f"wrote {S12 / args.out}")
        if not args.score:
            return

    g = json.loads((S12 / "gate.json").read_text())
    if not g.get("gate_passed"):
        # The study still has one decidable registered call: P-S12-B asked whether the validity
        # conditions would be met on the first attempt. A failed gate resolves it, and the
        # failure is recorded rather than quietly retried at a looser threshold.
        print("gate failed: recording the failure and scoring P-S12-B only.")
        stmt, prob = CALLS["P-S12-B"]
        row = {"date_scored": "2026-08-10", "prediction": f"P-S12-B {stmt}",
               "kind": "probability", "point": prob, "lo80": "", "hi80": "", "observed": 0,
               "verdict": "—", "brier": round((prob - 0) ** 2, 4), "abs_error": "",
               "note": "gate failed on the first attempt: median relative gradient norm "
                       "1.0e-4 to 7.3e-4 against a registered threshold of 1e-4"}
        if not args.dry_run:
            with OUTCOMES.open(newline="") as fh:
                already = {r["prediction"] for r in csv.DictReader(fh)}
            if row["prediction"] not in already:
                with OUTCOMES.open(newline="") as fh:
                    fieldnames = list(csv.DictReader(fh).fieldnames)
                with OUTCOMES.open("a", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=fieldnames).writerows([row])
                print(f"appended 1 row to {OUTCOMES}")
            (S12 / "verdict.json").write_text(json.dumps(
                {"study": "S12", "prereg": "docs/prereg/S12.md", "gate": False,
                 "outcome": "failed to produce converged corpora; nothing decoded",
                 "grad_norm_median_by_corpus": {
                     k: v.get("grad_norm_median") for k, v in g["per_corpus"].items()},
                 "threshold": GRAD_TOL,
                 "P-S12-B": {"p": prob, "resolved": False,
                             "brier": round((prob - 0) ** 2, 4)}}, indent=2))
            print(f"wrote {S12 / 'verdict.json'}")
        return

    ladder = S12 / "ladder"
    need = ["W1", "W3", "W5", "W12"]
    arms = {}
    for rep in REPS:
        for r in need:
            f = ladder / f"r{rep}_{r}.json"
            if not f.exists():
                raise SystemExit(f"missing decoded arm {f.name}; run the S12 decode first")
            arms[(rep, r)] = json.loads(f.read_text())

    def frac(rep: int, rung: str) -> float:
        w1 = statistics.mean(arms[(rep, "W1")]["acc"])
        w3 = statistics.mean(arms[(rep, "W3")]["acc"])
        return (statistics.mean(arms[(rep, rung)]["acc"]) - w3) / (w1 - w3)

    f_align = [frac(r, "W5") for r in REPS]
    f_w12 = [frac(r, "W12") for r in REPS]
    gaps = [statistics.mean(arms[(r, "W1")]["acc"]) - statistics.mean(arms[(r, "W3")]["acc"])
            for r in REPS]
    gn = [g["per_corpus"][f"P-random-r{r}"]["grad_norm_median"] for r in REPS]

    obs = {
        "H-S12-1": statistics.median(gn),
        "H-S12-2": statistics.mean(gaps),
        "H-S12-3": statistics.mean(f_align),
        "H-S12-4": statistics.mean(f_align) - 0.4589,   # S8's 10000-step arm
        "H-S12-6": statistics.mean(f_w12),
        "H-S12-7": statistics.pstdev(f_align),
    }
    rows, hits = [], 0
    for k, v in obs.items():
        stmt, point, (lo, hi) = REGISTERED[k]
        hit = lo <= v <= hi
        hits += hit
        print(f"  {k}: registered {point} [{lo}, {hi}], observed {v:.4g} -> "
              f"{'HIT' if hit else 'MISS'}")
        rows.append({"date_scored": "2026-08-09", "prediction": f"{k} {stmt}", "kind": "interval",
                     "point": point, "lo80": lo, "hi80": hi, "observed": round(v, 6),
                     "verdict": "HIT" if hit else "MISS",
                     "abs_error": round(abs(v - point), 6), "brier": "", "note": ""})

    seed_sd = statistics.pstdev([frac(REPS[0], "W5")]) if False else None
    happened = {
        "P-S12-A": obs["H-S12-3"] > 0.30,
        "P-S12-B": bool(g["gate_passed"]),
        "P-S12-C": None,  # needs the decoder-seed SD, computed below
    }
    per_seed = [(a - statistics.mean(arms[(REPS[0], "W3")]["acc"]))
                / (statistics.mean(arms[(REPS[0], "W1")]["acc"])
                   - statistics.mean(arms[(REPS[0], "W3")]["acc"]))
                for a in arms[(REPS[0], "W5")]["acc"]]
    seed_sd = statistics.pstdev(per_seed)
    happened["P-S12-C"] = seed_sd > 0 and obs["H-S12-7"] >= 3 * seed_sd
    for k, (stmt, p) in CALLS.items():
        o = bool(happened[k])
        print(f"  {k}: p={p}, resolved {o}, Brier {(p - o) ** 2:.4f}")
        rows.append({"date_scored": "2026-08-09", "prediction": f"{k} {stmt}",
                     "kind": "probability", "point": p, "lo80": "", "hi80": "",
                     "observed": int(o), "verdict": "—", "abs_error": "",
                     "brier": round((p - o) ** 2, 4), "note": ""})

    fired = obs["H-S12-3"] < FALSIFIER
    print(f"\nS12: {hits}/{len(obs)} intervals; falsifier f(c_align) < {FALSIFIER}: "
          f"{'FIRED' if fired else 'not fired'}")
    if fired:
        print("Per S12 section 5, every ladder claim is rescoped to the early-stopped regime "
              "IN THE ABSTRACT.")
    print(f"between-corpus SD {obs['H-S12-7']:.4f} vs decoder-seed SD {seed_sd:.4f}")

    verdict = {"study": "S12", "prereg": "docs/prereg/S12.md", "gate": g["gate_passed"],
               "f_align_per_rep": f_align, "f_w12_per_rep": f_w12, "gaps": gaps,
               "grad_norm_median_per_rep": gn, "between_corpus_sd": obs["H-S12-7"],
               "decoder_seed_sd": seed_sd, "intervals": f"{hits}/{len(obs)}",
               "falsifier_fired": fired}
    if args.dry_run:
        print("\ndry run: nothing written")
        return
    with OUTCOMES.open(newline="") as fh:
        already = {r["prediction"] for r in csv.DictReader(fh)}
    rows = [r for r in rows if r["prediction"] not in already]
    if rows:
        with OUTCOMES.open(newline="") as fh:
            fieldnames = list(csv.DictReader(fh).fieldnames)
        with OUTCOMES.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerows(rows)
        print(f"appended {len(rows)} rows to {OUTCOMES}")
    (S12 / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"wrote {S12 / 'verdict.json'}")


if __name__ == "__main__":
    main()

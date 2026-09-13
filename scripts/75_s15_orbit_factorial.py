#!/usr/bin/env python3
"""S15: the orbit-only intervention as a 2^4 factorial over sigma, rho, tau and pi.

docs/prereg/S15.md. S6 added the generators in one nested order, which cannot attribute the damage;
this decodes every non-empty factor set, reports Shapley values, replicates the full cell under two
more group draws, and adds a non-group scramble that changes the function.

    .venv/bin/python scripts/75_s15_orbit_factorial.py --dataset mnist
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sirengap.eval.decoder import train_matched_mlp  # noqa: E402
from sirengap.eval.rungs import CorpusCache, probe_coords  # noqa: E402
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
from sirengap.symmetry.dinf import GroupElement, apply  # noqa: E402

S6 = __import__("37_orbit_intervention")

FACTORS = ("sigma", "rho", "tau", "pi")
SEEDS = 5
HALF_RANGE = 5          # even windings 2m, m ~ U{-5..5}, so |j| <= 11 with the parity bit
MAIN_GROUP_SEED = 1234
REPLICATE_SEEDS = (5678, 9012)
GAP_TOL = 1e-4


def sample_factorial(params: SirenParams, gen: torch.Generator, on: dict[str, bool]) -> GroupElement:
    ds, js, perms = [], [], []
    for w, _ in params.hidden:
        b, n = w.shape[0], w.shape[1]
        zeros = torch.zeros(b, n, dtype=torch.long)
        d = torch.randint(0, 2, (b, n), generator=gen) if on["sigma"] else zeros
        r = torch.randint(0, 2, (b, n), generator=gen) if on["rho"] else zeros
        m = torch.randint(-HALF_RANGE, HALF_RANGE + 1, (b, n), generator=gen) if on["tau"] else zeros
        ds.append(d)
        js.append(2 * m + r)
        perms.append(torch.argsort(torch.rand(b, n, generator=gen), dim=1) if on["pi"]
                     else torch.arange(n).expand(b, n).clone())
    return GroupElement(d=tuple(ds), j=tuple(js), perm=tuple(perms))


def non_group_scramble(p: SirenParams, gen: torch.Generator) -> SirenParams:
    """Permute first-layer neurons without re-wiring W2's columns: the function changes."""
    (w1, b1), rest = p.hidden[0], p.hidden[1:]
    perm = torch.argsort(torch.rand(w1.shape[0], w1.shape[1], generator=gen), dim=1)
    w1s = torch.gather(w1, 1, perm.unsqueeze(-1).expand_as(w1))
    b1s = torch.gather(b1, 1, perm)
    return SirenParams(hidden=((w1s, b1s),) + tuple(rest), w_out=p.w_out, b_out=p.b_out)


def subset(by_split: dict[str, SirenParams], labels: dict, limit: int):
    if limit <= 0:
        return by_split, labels
    idx = {s: torch.arange(min(limit, p.batch)) for s, p in by_split.items()}
    sub = {s: SirenParams(hidden=tuple((w[idx[s]], b[idx[s]]) for w, b in p.hidden),
                          w_out=p.w_out[idx[s]], b_out=p.b_out[idx[s]])
           for s, p in by_split.items()}
    return sub, {s: labels[s][idx[s]] for s in labels}


def decode(params: dict[str, SirenParams], labels: dict, seeds: int, device: str) -> list[float]:
    feats = {s: p.flat() for s, p in params.items()}
    return [train_matched_mlp(feats, labels, seed=s, device=device).test_acc for s in range(seeds)]


def summarize(accs: list[float]) -> dict:
    a = np.array(accs)
    return {"acc": accs, "mean": float(a.mean()), "ci95": bootstrap_ci_mean(a)}


def shapley(delta: dict[frozenset, float]) -> dict[str, float]:
    n = len(FACTORS)
    out = {}
    for f in FACTORS:
        others = [g for g in FACTORS if g != f]
        total = 0.0
        for k in range(len(others) + 1):
            for s in itertools.combinations(others, k):
                s = frozenset(s)
                weight = math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n)
                total += weight * (delta[s | {f}] - delta[s])
        out[f] = total
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--protocol", default="P-shared-det")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N INRs per split")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, labels = subset(*cache.split_params(args.protocol), args.limit)
    probes = probe_coords(args.dataset)
    out: dict = {"study": "S15 factorial orbit intervention", "prereg": "docs/prereg/S15.md",
                 "dataset": args.dataset, "protocol": args.protocol, "seeds": args.seeds,
                 "limit": args.limit, "half_range": HALF_RANGE, "cells": {}}

    base = summarize(decode(by_split, labels, args.seeds, args.device))
    out["baseline"] = base
    print(f"baseline {base['mean']:.2f}", flush=True)

    def run_cell(name: str, scattered: dict[str, SirenParams], function_preserving: bool) -> float:
        t0 = time.time()
        gap = float(S6.max_functional_gap(by_split["test"], scattered["test"], probes))
        if function_preserving and gap > GAP_TOL:
            raise SystemExit(f"void cell {name}: functional gap {gap:.2e} > {GAP_TOL}")
        r = summarize(decode(scattered, labels, args.seeds, args.device))
        r.update({"delta": base["mean"] - r["mean"], "functional_gap": gap,
                  "wallclock_s": time.time() - t0})
        out["cells"][name] = r
        print(f"  {name:24s} acc {r['mean']:6.2f}  delta {r['delta']:+6.2f}  gap {gap:.1e}",
              flush=True)
        return r["delta"]

    delta: dict[frozenset, float] = {frozenset(): 0.0}
    for mask in itertools.product((0, 1), repeat=len(FACTORS)):
        if not any(mask):
            continue
        on = dict(zip(FACTORS, map(bool, mask)))
        name = "+".join(f for f in FACTORS if on[f])
        gen = torch.Generator().manual_seed(MAIN_GROUP_SEED)
        scattered = {s: apply(sample_factorial(p, gen, on), p) for s, p in by_split.items()}
        delta[frozenset(f for f in FACTORS if on[f])] = run_cell(name, scattered, True)

    full = dict.fromkeys(FACTORS, True)
    for seed in REPLICATE_SEEDS:
        gen = torch.Generator().manual_seed(seed)
        scattered = {s: apply(sample_factorial(p, gen, full), p) for s, p in by_split.items()}
        run_cell(f"full_draw{seed}", scattered, True)

    gen = torch.Generator().manual_seed(MAIN_GROUP_SEED)
    scrambled = {s: non_group_scramble(p, gen) for s, p in by_split.items()}
    run_cell("non_group_scramble", scrambled, False)

    phi = shapley(delta)
    out["shapley"] = phi
    out["shapley_sum"] = sum(phi.values())
    out["full_draws"] = [out["cells"]["sigma+rho+tau+pi"]["delta"]] + [
        out["cells"][f"full_draw{s}"]["delta"] for s in REPLICATE_SEEDS]
    print("shapley:", {k: round(v, 2) for k, v in phi.items()}, "sum", round(out["shapley_sum"], 2))

    suffix = "_SMOKE" if args.limit else ""
    path = ROOT / "results" / "s15" / f"factorial_{args.dataset}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

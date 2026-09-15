#!/usr/bin/env python3
"""S19 arm A3: partial scatter (sign and relabelling doses) and a layer-two non-group control.

docs/prereg/S19.md. The orbit intervention of S6/S15 always draws the sign and the relabelling at
full entropy, so it cannot show how the frozen decoder degrades as scatter grows. Here each is
applied at dose p: a sign dose gives every hidden neuron a fresh random sign with probability p,
and a relabelling dose permutes a random subset of round(p n) neurons of each hidden layer among
themselves, re-wiring consistently. p = 1 reproduces the S15 single-generator cells. The control
permutes the rows of W2 and the entries of b2 without re-wiring W3: it changes the function and
touches exactly the coordinates a layer-two relabelling touches.

    .venv/bin/python scripts/80_s19_dose_response.py --dataset mnist
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sirengap.eval.rungs import CorpusCache, probe_coords  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
from sirengap.symmetry.dinf import GroupElement, apply  # noqa: E402

S15 = __import__("75_s15_orbit_factorial")
S6 = __import__("37_orbit_intervention")

DOSES = (0.1, 0.25, 0.5, 1.0)
GROUP_SEED = 1234
GAP_TOL = 1e-4


def _identity(b: int, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zeros = torch.zeros(b, n, dtype=torch.long)
    return zeros, zeros.clone(), torch.arange(n).expand(b, n).clone()


def sign_dose(p: SirenParams, gen: torch.Generator, q: float) -> GroupElement:
    """Each hidden neuron receives d ~ Bernoulli(1/2) with probability q, else d = 0."""
    ds, js, perms = [], [], []
    for w, _ in p.hidden:
        b, n = w.shape[0], w.shape[1]
        d, j, perm = _identity(b, n)
        hit = torch.rand(b, n, generator=gen) < q
        d = torch.randint(0, 2, (b, n), generator=gen) * hit.long()
        ds.append(d)
        js.append(j)
        perms.append(perm)
    return GroupElement(d=tuple(ds), j=tuple(js), perm=tuple(perms))


def relabel_dose(p: SirenParams, gen: torch.Generator, q: float) -> GroupElement:
    """Permute a uniformly chosen subset of round(q n) neurons of each layer among itself."""
    ds, js, perms = [], [], []
    for w, _ in p.hidden:
        b, n = w.shape[0], w.shape[1]
        d, j, perm = _identity(b, n)
        k = int(round(q * n))
        if k >= 2:
            chosen = torch.argsort(torch.rand(b, n, generator=gen), dim=1)[:, :k]
            shuffle = torch.argsort(torch.rand(b, k, generator=gen), dim=1)
            perm.scatter_(1, chosen, torch.gather(chosen, 1, shuffle))
        ds.append(d)
        js.append(j)
        perms.append(perm)
    return GroupElement(d=tuple(ds), j=tuple(js), perm=tuple(perms))


def layer2_scramble(p: SirenParams, gen: torch.Generator) -> SirenParams:
    """Permute layer-two neurons (rows of W2, entries of b2) without re-wiring W3: not a group
    element, so the function changes."""
    (w1, b1), (w2, b2) = p.hidden
    perm = torch.argsort(torch.rand(w2.shape[0], w2.shape[1], generator=gen), dim=1)
    w2s = torch.gather(w2, 1, perm.unsqueeze(-1).expand_as(w2))
    b2s = torch.gather(b2, 1, perm)
    return SirenParams(hidden=((w1, b1), (w2s, b2s)), w_out=p.w_out, b_out=p.b_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--protocol", default="P-shared-det")
    ap.add_argument("--seeds", type=int, default=S15.SEEDS)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N INRs per split")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, labels = S15.subset(*cache.split_params(args.protocol), args.limit)
    probes = probe_coords(args.dataset)
    out: dict = {"study": "S19 A3 dose response", "prereg": "docs/prereg/S19.md",
                 "dataset": args.dataset, "protocol": args.protocol, "seeds": args.seeds,
                 "limit": args.limit, "doses": list(DOSES), "cells": {}}

    base = S15.summarize(S15.decode(by_split, labels, args.seeds, args.device))
    out["baseline"] = base
    print(f"baseline {base['mean']:.2f}", flush=True)

    def run_cell(name: str, scattered: dict[str, SirenParams], function_preserving: bool) -> None:
        t0 = time.time()
        gap = float(S6.max_functional_gap(by_split["test"], scattered["test"], probes))
        if function_preserving and gap > GAP_TOL:
            raise SystemExit(f"void cell {name}: functional gap {gap:.2e} > {GAP_TOL}")
        r = S15.summarize(S15.decode(scattered, labels, args.seeds, args.device))
        r.update({"delta": base["mean"] - r["mean"], "functional_gap": gap,
                  "wallclock_s": time.time() - t0})
        out["cells"][name] = r
        print(f"  {name:22s} acc {r['mean']:6.2f}  delta {r['delta']:+6.2f}  gap {gap:.1e}",
              flush=True)

    for kind, fn in (("sign", sign_dose), ("relabel", relabel_dose)):
        for q in DOSES:
            gen = torch.Generator().manual_seed(GROUP_SEED)
            scattered = {s: apply(fn(p, gen, q), p) for s, p in by_split.items()}
            run_cell(f"{kind}_p{q}", scattered, True)

    gen = torch.Generator().manual_seed(GROUP_SEED)
    run_cell("layer2_scramble", {s: layer2_scramble(p, gen) for s, p in by_split.items()}, False)

    deltas = {k: [out["cells"][f"{k}_p{q}"]["delta"] for q in DOSES] for k in ("sign", "relabel")}
    out["monotone"] = {k: all(a <= b for a, b in zip(v, v[1:])) for k, v in deltas.items()}
    suffix = "_SMOKE" if args.limit else ""
    path = ROOT / "results" / "s19" / f"dose_response_{args.dataset}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

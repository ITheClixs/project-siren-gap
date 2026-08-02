#!/usr/bin/env python3
"""S6 — the orbit-only intervention: symmetry sensitivity, measured rather than inferred.

Every earlier rung inferred a "symmetry share" from the shared- versus random-initialization gap.
That inference does not hold. An orbit-valued map preserves the function, but it can still route
an orbit-invariant property into a coordinate the reader finds easy: take tau_k with
k = M * y(theta) for an invariant binary y and a large M. Nothing about the function changes, yet
a trivial reader now predicts y. So "no information was created" does not license "the gain
measures nuisance removal", and the recovery fraction is an *algorithm-relative recoverable
fraction*, not a causal share.

This study measures the causal quantity directly, by intervening on the group instead of on the
initialization:

    take a corpus with NO nuisance (P-shared-det, every INR fitted from the same init),
    hold each fitted network and its function fixed,
    apply an independently sampled g_i ~ mu_B to each one,
    and measure what that does to a reader.

Because the same networks and the same functions appear on both sides, the degradation is caused
by the group action and by nothing else. Then each treatment (c_sort, c_align, the invariant
encoding, the equivariant readers) is scored on how much of *that* it recovers.

There is no uniform probability measure on the infinite group D_infinity, so mu is a *family*:
windings j ~ Uniform{-B..B}, signs d ~ Bernoulli(1/2), permutations uniform on S_n, and every
result is reported as a function of B rather than at one arbitrary truncation.

Usage:
  .venv/bin/python scripts/37_orbit_intervention.py --dataset mnist --windings 0 1 3 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.canon.calign import c_align  # noqa: E402
from sirengap.canon.csort import c_sort  # noqa: E402
from sirengap.canon.deep_invariants import encode_deep  # noqa: E402
from sirengap.eval.decoder import train_matched_mlp  # noqa: E402
from sirengap.eval.rungs import (  # noqa: E402
    SPLITS,
    CorpusCache,
    Rung,
    _chunked,
    probe_coords,
    shared_init_template,
)
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402
from sirengap.models.forward import max_functional_gap  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
from sirengap.symmetry.dinf import GroupElement, apply  # noqa: E402

SEEDS = 5


def sample_mu(params: SirenParams, generator: torch.Generator, windings: int,
              permute: bool = True) -> GroupElement:
    """mu_B: j ~ U{-B..B}, d ~ Bernoulli(1/2), permutation uniform (or identity)."""
    ds, js, perms = [], [], []
    for w, _ in params.hidden:
        b, n = w.shape[0], w.shape[1]
        ds.append(torch.randint(0, 2, (b, n), generator=generator))
        if windings == 0:
            js.append(torch.zeros(b, n, dtype=torch.long))
        else:
            js.append(torch.randint(-windings, windings + 1, (b, n), generator=generator))
        perms.append(
            torch.argsort(torch.rand(b, n, generator=generator), dim=1) if permute
            else torch.arange(n).expand(b, n).clone()
        )
    return GroupElement(d=tuple(ds), j=tuple(js), perm=tuple(perms))


def scatter_corpus(by_split: dict[str, SirenParams], windings: int, seed: int,
                   permute: bool = True) -> dict[str, SirenParams]:
    """Apply an independent group element to every INR. Functions are unchanged by construction."""
    gen = torch.Generator().manual_seed(seed)
    return {s: apply(sample_mu(p, gen, windings, permute), p) for s, p in by_split.items()}


def treatments(scattered: dict[str, SirenParams], dataset: str,
               template: SirenParams, probes: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    """The feature maps under test, all applied to the *same* scattered corpus."""
    out = {"raw": {s: p.flat() for s, p in scattered.items()}}
    out["c_sort"] = {s: _chunked(lambda q: c_sort(q)[0].flat(), p) for s, p in scattered.items()}
    out["c_align"] = {
        s: _chunked(lambda q: c_align(q, template, probes)[0].flat(), p)
        for s, p in scattered.items()
    }
    out["invariants"] = {s: _chunked(encode_deep, p) for s, p in scattered.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--protocol", default="P-shared-det",
                    help="must be a corpus with no initialization nuisance")
    ap.add_argument("--windings", nargs="+", type=int, default=[0, 1, 3, 10])
    ap.add_argument("--no-permute", action="store_true",
                    help="isolate the D_infinity part by fixing the permutation to identity")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, labels = cache.split_params(args.protocol)
    template = shared_init_template(by_split["train"])
    probes = probe_coords(args.dataset)

    out: dict = {
        "study": "S6 orbit-only intervention",
        "prereg": "docs/prereg/S6.md",
        "dataset": args.dataset, "protocol": args.protocol,
        "permute": not args.no_permute,
        "note": "same fitted networks and same functions on both sides; only the group acts",
        "baseline": {}, "by_winding": {},
    }

    # baseline: the untouched corpus, which is the no-nuisance ceiling
    base = Rung("raw", {s: p.flat() for s, p in by_split.items()}, labels)
    base_accs = [train_matched_mlp(base.feats, labels, seed=s, device=args.device).test_acc
                 for s in range(args.seeds)]
    out["baseline"] = {"acc": base_accs, "mean": float(np.mean(base_accs)),
                       "ci95": bootstrap_ci_mean(np.array(base_accs))}
    print(f"baseline (no scatter): {np.mean(base_accs):.2f}", flush=True)

    for B in args.windings:
        t0 = time.time()
        scattered = scatter_corpus(by_split, B, seed=1234, permute=not args.no_permute)
        gap = max_functional_gap(by_split["test"], scattered["test"], probes)
        feats = treatments(scattered, args.dataset, template, probes)
        cell: dict = {"functional_gap": float(gap), "treatments": {}}
        for name, f in feats.items():
            accs = [train_matched_mlp(f, labels, seed=s, device=args.device).test_acc
                    for s in range(args.seeds)]
            a = np.array(accs)
            cell["treatments"][name] = {
                "acc": accs, "mean": float(a.mean()), "ci95": bootstrap_ci_mean(a),
            }
            print(f"  B={B:3d} {name:12s} {a.mean():6.2f}", flush=True)
        raw = cell["treatments"]["raw"]["mean"]
        drop = out["baseline"]["mean"] - raw
        cell["delta_sym"] = drop
        for name, r in cell["treatments"].items():
            r["recovered_fraction"] = (r["mean"] - raw) / drop if abs(drop) > 1e-9 else float("nan")
        cell["wallclock_s"] = time.time() - t0
        out["by_winding"][str(B)] = cell
        print(f"  B={B:3d} delta_sym = {drop:+.2f} pts   (functional gap {gap:.2e})", flush=True)

    path = ROOT / "results" / "s6" / f"orbit_{args.dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Is c_align a canonicalizer on *production* corpora, or only on random parameters?

T3 asserts c(g theta) = c(theta) on randomly drawn parameters. The external review asked
the sharper version: does it hold on fitted INRs, whose first-layer directions come within
3e-4 rad of the parallel-frequency stratum, so that the sort keys are nearly tied?

This script measures the residual directly on a sample of the real corpus, for both
canonicalizers, over random group elements. It also reports the share of INRs whose residual
exceeds a tolerance, since a canonicalizer that works for 99% of a corpus and fails for 1% is
a different object from one that works everywhere.

Runs on CPU by default so it does not contend for the accelerator.

Usage:
  .venv/bin/python scripts/42_canon_equivariance_audit.py --dataset mnist --n 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sirengap.canon.calign import c_align  # noqa: E402
from sirengap.canon.csort import c_sort  # noqa: E402
from sirengap.eval.rungs import CorpusCache, probe_coords, shared_init_template  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
from sirengap.symmetry.dinf import apply, random_element  # noqa: E402

TOL = 1e-4  # the tolerance T3 asserts on random parameters


def _index(params: SirenParams, idx: torch.Tensor) -> SirenParams:
    return SirenParams(
        hidden=tuple((w[idx], b[idx]) for w, b in params.hidden),
        w_out=params.w_out[idx], b_out=params.b_out[idx],
    )


def per_inr_residual(a: SirenParams, b: SirenParams) -> np.ndarray:
    """Max absolute coordinate disagreement per INR, relative to that INR's scale."""
    fa, fb = a.flat(), b.flat()
    scale = fa.abs().amax(dim=1).clamp_min(1e-12)
    return ((fa - fb).abs().amax(dim=1) / scale).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--protocol", default="P-random")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--max-windings", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default="data/inrbench")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, _ = cache.split_params(args.protocol)
    full = by_split["test"]
    idx = torch.arange(min(args.n, full.batch))
    params = _index(full, idx)
    template = shared_init_template(by_split["train"])
    probes = probe_coords(args.dataset)

    canon = {
        "c_sort": c_sort(params)[0],
        "c_align": c_align(params, template, probes)[0],
    }

    report: dict = {
        "dataset": args.dataset, "protocol": args.protocol, "n_inrs": int(len(idx)),
        "trials": args.trials, "max_windings": args.max_windings, "tolerance": TOL,
        "note": "residual of c(g theta) against c(theta), per INR, relative to that INR's scale",
        "canonicalizers": {},
    }

    for name, base in canon.items():
        res = []
        for trial in range(args.trials):
            gen = torch.Generator().manual_seed(4200 + trial)
            moved = apply(random_element(params, gen, max_windings=args.max_windings), params)
            got = c_sort(moved)[0] if name == "c_sort" else c_align(moved, template, probes)[0]
            res.append(per_inr_residual(base, got))
        r = np.concatenate(res)
        report["canonicalizers"][name] = {
            "median": float(np.median(r)),
            "p95": float(np.percentile(r, 95)),
            "max": float(r.max()),
            "share_above_tol": float((r > TOL).mean()),
        }
        print(f"{name:8s} median {np.median(r):.2e}  p95 {np.percentile(r, 95):.2e}  "
              f"max {r.max():.2e}  above {TOL:g}: {(r > TOL).mean():.1%}")

    out = ROOT / "results" / "audits"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"canon_equivariance_{args.dataset}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

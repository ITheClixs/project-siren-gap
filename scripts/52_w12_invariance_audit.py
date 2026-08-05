#!/usr/bin/env python3
"""Is W12 invariant on *production* INRs, or only on freshly initialised ones?

T16 asserts invariance on parameters from `init_from_seeds`. Fitted networks are a harder
case: their weight scales differ by orders of magnitude across layers, and the corpora pass
within 3e-4 rad of the parallel-frequency stratum. Invariance is algebraic and should not care,
but "should not care" is what an audit is for -- and a rung claiming 0.92 recovery deserves the
check before anything is built on it.

Invariance is a property of the architecture and the feature map, not of the trained weights,
so a randomly initialised reader is the right instrument. Runs on CPU by default.

Usage:
  .venv/bin/python scripts/52_w12_invariance_audit.py --dataset mnist --n 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sirengap.eval.rungs import CorpusCache  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
from sirengap.models.phasor import (  # noqa: E402
    PhasorGradedReader,
    apply_scale,
    feature_scale,
    phasor_features,
)
from sirengap.symmetry.dinf import apply, random_element  # noqa: E402

TOL = 1e-4  # the tolerance T16 passes at on synthetic parameters


def _index(params: SirenParams, idx: torch.Tensor) -> SirenParams:
    return SirenParams(
        hidden=tuple((w[idx], b[idx]) for w, b in params.hidden),
        w_out=params.w_out[idx], b_out=params.b_out[idx],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--protocol", default="P-random")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--windings", nargs="+", type=int, default=[3, 10, 40])
    ap.add_argument("--width", type=int, default=186)
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--ungraded", action="store_true",
                    help="audit the matched control instead: how far from invariant is it really?")
    ap.add_argument("--raw-bias", action="store_true",
                    help="audit the third arm (W12b): same graded skeleton, unlifted bias, which "
                         "must move under a winding or the control is vacuous (S10 section 5)")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, _ = cache.split_params(args.protocol)
    full = by_split["test"]
    params = _index(full, torch.arange(min(args.n, full.batch)))

    base = phasor_features(params, raw_bias=args.raw_bias)
    stats = feature_scale(base)

    torch.manual_seed(0)
    model = PhasorGradedReader.from_features(base, width=args.width,
                                             graded=not args.ungraded)
    model.eval()
    with torch.no_grad():
        logits = model(apply_scale(base, stats))

    report: dict = {
        "dataset": args.dataset, "protocol": args.protocol,
        "n_inrs": int(params.batch), "width": args.width, "tolerance": TOL,
        "graded": not args.ungraded, "raw_bias": bool(args.raw_bias),
        "note": "logit move under group elements, on fitted networks; a random reader, since "
                "invariance is a property of the architecture and not of trained weights",
        "by_winding": {},
    }
    for b in args.windings:
        moves, feat_moves = [], []
        for trial in range(args.trials):
            gen = torch.Generator().manual_seed(900 + trial)
            g = random_element(params, gen, max_windings=b)
            moved = phasor_features(apply(g, params), raw_bias=args.raw_bias)
            with torch.no_grad():
                out = model(apply_scale(moved, stats))
            scale = logits.abs().amax(dim=1).clamp_min(1e-12)
            moves.append(float(((logits - out).abs().amax(dim=1) / scale).max()))
            # The neutral block is permutation-*covariant*, not invariant: a permutation
            # reorders its rows, so comparing it elementwise after one is meaningless. Measure
            # it under an identity-permutation element instead, where it must be exactly fixed.
            g_dinf = random_element(params, gen, max_windings=b, identity_perm=True)
            dinf_only = phasor_features(apply(g_dinf, params), raw_bias=args.raw_bias)
            fm = max(
                float((base[layer][(0, 0)] - dinf_only[layer][(0, 0)]).abs().max()
                      / base[layer][(0, 0)].abs().max().clamp_min(1e-12))
                for layer in ("l1", "l2")
            )
            feat_moves.append(fm)
        report["by_winding"][str(b)] = {
            "logit_move_max": max(moves),
            "neutral_block_move_max_dinf_only": max(feat_moves),
            "within_tolerance": bool(max(moves) < TOL),
        }
        print(f"|j|<={b:3d}: logits move {max(moves):.2e}, neutral blocks (D_inf only) {max(feat_moves):.2e} "
              f"-> {'OK' if max(moves) < TOL else 'FAIL'}")

    report["all_within_tolerance"] = all(
        v["within_tolerance"] for v in report["by_winding"].values()
    )
    out_dir = ROOT / "results" / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "w12b" if args.raw_bias else "w12u" if args.ungraded else "w12"
    path = out_dir / f"{tag}_invariance_{args.dataset}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")
    if args.ungraded or args.raw_bias:
        # the control is *supposed* to fail; the number is the point, not the verdict
        return
    if not report["all_within_tolerance"]:
        raise SystemExit("W12 is not invariant on production INRs -- the rung is void (S9 section 4)")


if __name__ == "__main__":
    main()

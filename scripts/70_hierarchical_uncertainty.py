#!/usr/bin/env python3
"""Uncertainty over evaluation items as well as decoder seeds.

Every interval in the ladder resamples decoder seeds. That resolves one level of a hierarchy
whose other levels are datasets, images, fit realizations and seeds, and reporting only the
innermost level understates how much a number could move. This script adds the evaluation-item
level, which is the one that can be recovered without refitting a corpus: each rung is decoded
as usual, the per-item correctness of the selected model is retained, and the recoverable
fraction is then resampled over items and seeds jointly.

What this does not resolve is the fit-realization level. Two fits of the same image from
different initializations are different draws, and separating that variance needs refits rather
than a different analysis of these runs. The paper says so rather than implying otherwise.

Usage:
  .venv/bin/python scripts/70_hierarchical_uncertainty.py --dataset mnist
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sirengap.eval.decoder import train_matched_mlp  # noqa: E402
from sirengap.eval.rungs import CorpusCache  # noqa: E402

_ladder = __import__("11_ladder")
build_rung, PROTO = _ladder.build_rung, _ladder.PROTO

# The anchors and the exact treatments. W1 and W3 set the scale; the rest are read against it.
RUNGS = ["W1", "W3", "W4", "W5", "W10", "W10c"]


def decode(rung, seeds, device):
    """Per-item correctness of the selected model, one row per decoder seed."""
    rows, accs = [], []
    for s in range(seeds):
        res = train_matched_mlp(
            rung.feats, rung.labels, seed=s, device=device,
            augment=rung.augment, params_train=rung.params_train, flatten=rung.flatten,
        )
        rows.append(res.test_correct.numpy().astype(bool))
        accs.append(res.test_acc)
        print(f"    seed {s}: {res.test_acc:.2f}", flush=True)
    return np.stack(rows), np.array(accs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="results/hierarchical")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    hits, accs, labels = {}, {}, {}
    for name in RUNGS:
        print(f"  {name}", flush=True)
        rung = build_rung(name, cache, args.dataset, args.device)
        hits[name], accs[name] = decode(rung, args.seeds, args.device)
        labels[name] = rung.labels["test"].numpy()

    # Pairing over items is only meaningful if every rung scores the same test images in the
    # same order. The corpora differ in how they were fitted, not in which images they hold.
    ref = labels["W1"]
    for name in RUNGS:
        if labels[name].shape != ref.shape or not np.array_equal(labels[name], ref):
            raise SystemExit(f"test split of {name} is not aligned with W1; pairing invalid")
    n_items = len(ref)
    print(f"\n  {n_items} test items, aligned across all {len(RUNGS)} rungs")

    rng = np.random.default_rng(0)
    out = {"dataset": args.dataset, "items": int(n_items), "seeds": args.seeds,
           "draws": args.draws, "rungs": {}}

    def fraction(mat_k, mat_1, mat_3):
        num = mat_k.mean() - mat_3.mean()
        den = mat_1.mean() - mat_3.mean()
        return num / den if den else np.nan

    print(f"\n{'rung':>6s} {'mean f':>8s} {'seeds only':>22s} {'seeds + items':>24s}  widen")
    for name in RUNGS:
        if name in ("W1", "W3"):
            continue
        seed_only, both = [], []
        for _ in range(args.draws):
            s = rng.integers(0, args.seeds, args.seeds)
            seed_only.append(fraction(hits[name][s], hits["W1"][s], hits["W3"][s]))
            i = rng.integers(0, n_items, n_items)
            both.append(fraction(hits[name][s][:, i], hits["W1"][s][:, i], hits["W3"][s][:, i]))
        seed_only, both = np.array(seed_only), np.array(both)
        lo_s, hi_s = np.percentile(seed_only, [2.5, 97.5])
        lo_b, hi_b = np.percentile(both, [2.5, 97.5])
        point = fraction(hits[name], hits["W1"], hits["W3"])
        out["rungs"][name] = {
            "f": float(point),
            "ci_seeds": [float(lo_s), float(hi_s)],
            "ci_seeds_items": [float(lo_b), float(hi_b)],
            "width_seeds": float(hi_s - lo_s),
            "width_seeds_items": float(hi_b - lo_b),
            "acc_mean": float(accs[name].mean()),
        }
        print(f"{name:>6s} {point:8.3f} {f'[{lo_s:.3f}, {hi_s:.3f}]':>22s} "
              f"{f'[{lo_b:.3f}, {hi_b:.3f}]':>24s}  {(hi_b-lo_b)/(hi_s-lo_s):5.2f}x")

    out["anchors"] = {k: {"acc_mean": float(accs[k].mean()),
                          "acc": accs[k].tolist()} for k in ("W1", "W3")}
    d = Path(args.out); d.mkdir(parents=True, exist_ok=True)
    p = d / f"{args.dataset}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

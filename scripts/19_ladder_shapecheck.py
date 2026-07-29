#!/usr/bin/env python3
"""Instrument shape-check for the S1 ladder on a new dataset — no decoder, no accuracy.

Builds every rung's feature map and prints its shape, dtype and finiteness. This is the
pre-registration-safe smoke test: it establishes that the apparatus runs on a new corpus
without producing any number that could bias a registered prediction (cf. prereg S1
addendum 01, which used the same discipline on one shard).

Usage:
  .venv/bin/python scripts/19_ladder_shapecheck.py --dataset cifar10 --rungs P0 W3 W5 W10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sirengap.eval.rungs import SPLITS, CorpusCache  # noqa: E402

_ladder = __import__("11_ladder")
build_rung = _ladder.build_rung
ALL_RUNGS = _ladder.ALL_RUNGS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--rungs", nargs="+", default=["all"])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    args = ap.parse_args()

    wanted = list(ALL_RUNGS) if args.rungs == ["all"] else args.rungs
    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)

    for name in wanted:
        t0 = time.time()
        try:
            rung = build_rung(name, cache, args.dataset, args.device)
        except Exception as exc:  # noqa: BLE001 — a shape check reports, it does not raise
            print(f"{name:8s} FAILED  {type(exc).__name__}: {exc}", flush=True)
            continue
        shapes = {s: tuple(rung.feats[s].shape) for s in SPLITS}
        finite = all(torch.isfinite(rung.feats[s]).all().item() for s in SPLITS)
        n_lab = {s: int(rung.labels[s].shape[0]) for s in SPLITS}
        rows_match = all(shapes[s][0] == n_lab[s] for s in SPLITS)
        print(
            f"{name:8s} D={shapes['train'][1]:5d}  rows={ {s: shapes[s][0] for s in SPLITS} }  "
            f"labels_aligned={rows_match}  finite={finite}  "
            f"dtype={rung.feats['train'].dtype}  {time.time() - t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()

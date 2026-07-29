#!/usr/bin/env python3
"""EXPLORATORY: does W10's CIFAR-10 advantage come from having three output channels?

Not pre-registered. The S1-CIFAR registration predicted f(W10)_CIFAR > f(W10)_MNIST from the
encoding's algebra — with c = 3 each neuron's outgoing vector u_i in R^c carries strictly more
D_infty-visible structure than the c = 1 case — and that call resolved correctly (0.534 vs 0.269).
But CIFAR-10 also differs from the grayscale corpora in image statistics, so the channel mechanism
is confounded with signal complexity.

This ablation separates them *within* the CIFAR corpus and without any new fitting, by changing
only what the encoder is allowed to read from the fitted networks:

  full       all c = 3 output channels                     D = 384   (the registered W10)
  truncated  output channel 0 only                         D = 320   (matches the grayscale D)
  averaged   channels averaged into one                    D = 320   (same D, all 3 channels' info)

`truncated` and `averaged` have identical dimension, so the pair separates "more channels carry
more information" from "more feature dimensions let the decoder win". The networks themselves are
untouched; only the feature map changes, exactly as every other rung does.

Usage:
  .venv/bin/python scripts/25_w10_channel_ablation.py --dataset cifar10
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

from sirengap.canon.deep_invariants import encode_deep  # noqa: E402
from sirengap.eval.decoder import linear_probe, train_matched_mlp  # noqa: E402
from sirengap.eval.rungs import SPLITS, CorpusCache, Rung, _chunked  # noqa: E402
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402

SEEDS = 5


def truncate_channels(p: SirenParams) -> SirenParams:
    """Keep output channel 0 only — the encoder sees a c = 1 readout."""
    return SirenParams(hidden=p.hidden, w_out=p.w_out[:, :1, :], b_out=p.b_out[:, :1])


def average_channels(p: SirenParams) -> SirenParams:
    """Collapse the readout to its channel mean — same dimension as `truncate`, all 3 channels."""
    return SirenParams(
        hidden=p.hidden,
        w_out=p.w_out.mean(dim=1, keepdim=True),
        b_out=p.b_out.mean(dim=1, keepdim=True),
    )


ARMS = {
    "full_c3": lambda p: p,
    "truncated_c1": truncate_channels,
    "averaged_c1": average_channels,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, labels = cache.split_params("P-random")

    ladder = ROOT / "results" / "ladder" / args.dataset
    anchors = {}
    for r in ("W1", "W3"):
        anchors[r] = json.loads((ladder / f"{r}.json").read_text())["acc"]
    w1, w3 = np.array(anchors["W1"]), np.array(anchors["W3"])

    out = {
        "status": "EXPLORATORY — not pre-registered, not confirmatory evidence",
        "question": "is W10's CIFAR advantage the c=3 output channels, or the images?",
        "dataset": args.dataset,
        "W1": float(w1.mean()),
        "W3": float(w3.mean()),
        "seeds": SEEDS,
        "arms": {},
    }

    for name, fn in ARMS.items():
        t0 = time.time()
        feats = {s: _chunked(lambda p: encode_deep(fn(p)), by_split[s]) for s in SPLITS}
        rung = Rung(f"W10-{name}", feats, labels, notes=f"W10 channel ablation: {name}")
        accs = [
            train_matched_mlp(feats, labels, seed=s, device=args.device).test_acc
            for s in range(SEEDS)
        ]
        a = np.array(accs)
        n = min(len(a), len(w1), len(w3))
        f = (a[:n] - w3[:n]) / (w1[:n] - w3[:n])
        out["arms"][name] = {
            "feature_dim": int(feats["train"].shape[1]),
            "acc": accs,
            "mean": float(a.mean()),
            "ci95_bootstrap": bootstrap_ci_mean(a),
            "recovery_fraction": float(f.mean()),
            "f_ci95": bootstrap_ci_mean(f),
            "linear_probe": linear_probe(feats, labels, seed=0, device=args.device),
            "wallclock_s": time.time() - t0,
        }
        r = out["arms"][name]
        print(f"{name:14s} D={r['feature_dim']:4d}  acc={r['mean']:6.2f}  "
              f"f={r['recovery_fraction']:.4f}  probe={r['linear_probe']}", flush=True)
        del rung

    path = ladder / "EXPLORATORY_w10_channel_ablation.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

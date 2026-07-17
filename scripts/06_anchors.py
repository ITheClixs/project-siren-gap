#!/usr/bin/env python3
"""Replication anchors A1 and A2 (docs/REPLICATION.md; binding criteria fixed at G0/G1).

A1: acc(W1 = P-shared-det raw) - acc(W3 = P-random raw) > 10 pts, 95% CI excl. 0.
A2: acc(W6 = W3 + bounded group augmentation) - acc(W3) > 1 pt, CI excl. 0
    (perm + sigma + tau/rho with |j| <= 1; Shamsian's unbounded variant documented harmful).

Usage: .venv/bin/python scripts/06_anchors.py [--seeds 5] [--device mps] [--which A1 A2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sirengap.data.schema import load_corpus  # noqa: E402
from sirengap.eval.decoder import knn_accuracy, linear_probe, train_matched_mlp  # noqa: E402
from sirengap.eval.stats import bootstrap_ci_mean, paired_summary  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
from sirengap.symmetry.dinf import apply, random_element  # noqa: E402

ROOT = Path("data/inrbench/mnist")


def split_tensors(params: SirenParams, meta) -> tuple[dict, dict, dict]:
    feats, labels, sub_params = {}, {}, {}
    flat = params.flat()
    for split in ("train", "val", "test"):
        idx = torch.from_numpy((meta["split"] == split).to_numpy().nonzero()[0])
        feats[split] = flat[idx]
        labels[split] = torch.from_numpy(meta.loc[meta["split"] == split, "label"].to_numpy())
        sub_params[split] = SirenParams(
            hidden=tuple((w[idx], b[idx]) for w, b in params.hidden),
            w_out=params.w_out[idx],
            b_out=params.b_out[idx],
        )
    return feats, labels, sub_params


def bounded_aug(sub: SirenParams, gen: torch.Generator) -> SirenParams:
    return apply(random_element(sub, gen, max_windings=1), sub)


def run_rung(name, feats, labels, seeds, device, aug=None, params_train=None):
    accs, probes = [], {}
    for s in range(seeds):
        res = train_matched_mlp(
            feats, labels, seed=s, device=device,
            augment=aug, params_train=params_train,
            flatten=(lambda p: p.flat()) if aug else None,
        )
        accs.append(res.test_acc)
        print(f"  {name} seed {s}: test {res.test_acc:.2f} (val {res.val_acc:.2f}, {res.epochs_ran} ep)", flush=True)
    probes["linear_probe"] = linear_probe(feats, labels, seed=0, device=device)
    probes["knn10_cosine"] = knn_accuracy(feats, labels)
    return accs, probes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--which", nargs="+", default=["A1", "A2"])
    args = ap.parse_args()

    out: dict = {}
    p_w1, m_w1 = load_corpus(ROOT / "P-shared-det")
    p_w3, m_w3 = load_corpus(ROOT / "P-random")
    f1, l1, _ = split_tensors(p_w1, m_w1)
    f3, l3, sp3 = split_tensors(p_w3, m_w3)

    print("rung W3 (P-random raw):", flush=True)
    acc_w3, probes_w3 = run_rung("W3", f3, l3, args.seeds, args.device)

    if "A1" in args.which:
        print("rung W1 (P-shared-det raw):", flush=True)
        acc_w1, probes_w1 = run_rung("W1", f1, l1, args.seeds, args.device)
        s = paired_summary(np.array(acc_w1), np.array(acc_w3))
        out["A1"] = {
            "acc_W1": acc_w1, "acc_W3": acc_w3, "probes_W1": probes_w1, "probes_W3": probes_w3,
            "stats_W1_minus_W3": s,
            "boot_ci_W1": bootstrap_ci_mean(np.array(acc_w1)),
            "passes_binding": bool(s["mean_diff"] > 10.0 and s["ci95"][0] > 0),
        }
        print(json.dumps(out["A1"]["stats_W1_minus_W3"], indent=2), flush=True)

    if "A2" in args.which:
        print("rung W6 (W3 + bounded group aug):", flush=True)
        acc_w6, probes_w6 = run_rung(
            "W6", f3, l3, args.seeds, args.device,
            aug=bounded_aug, params_train=sp3["train"],
        )
        s = paired_summary(np.array(acc_w6), np.array(acc_w3))
        out["A2"] = {
            "acc_W6": acc_w6, "probes_W6": probes_w6, "stats_W6_minus_W3": s,
            "passes_binding": bool(s["mean_diff"] > 1.0 and s["ci95"][0] > 0),
        }
        print(json.dumps(out["A2"]["stats_W6_minus_W3"], indent=2), flush=True)

    dest = Path("results/anchors")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "anchors_mnist.json").write_text(json.dumps(out, indent=2))
    print("written results/anchors/anchors_mnist.json", flush=True)


if __name__ == "__main__":
    main()

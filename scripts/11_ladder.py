#!/usr/bin/env python3
"""S1 decomposition ladder (docs/prereg/S1.md) — the dissertation's spine, figure F9.

Builds each rung's feature map, decodes it with the frozen matched-MLP apparatus at the
pre-registered seed count for its noise class, and writes one JSON cell per rung. Rungs are
independent, so the script is resumable: an existing cell is skipped unless --force.

Usage:
  .venv/bin/python scripts/11_ladder.py --dataset mnist --rungs P0 P1 W1 W2 W3 W4 W5
  .venv/bin/python scripts/11_ladder.py --dataset mnist --rungs all
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
from sirengap.canon.calign import c_align  # noqa: E402
from sirengap.canon.csort import c_sort  # noqa: E402
from sirengap.canon.deep_invariants import encode_deep  # noqa: E402
from sirengap.eval.decoder import knn_accuracy, linear_probe, train_matched_mlp  # noqa: E402
from sirengap.eval.rungs import (  # noqa: E402
    SPLITS,
    CorpusCache,
    Rung,
    _chunked,
    _index,
    bounded_aug,
    build_frame,
    frame_average,
    probe_coords,
    render_features,
    shared_init_template,
)
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
from sirengap.symmetry.dinf import apply  # noqa: E402

# seed policy from the power memo (docs/prereg/S1.md §4)
SEEDS_DETERMINISTIC = 5
SEEDS_AUGMENTED = 15
ALL_RUNGS = ("P0", "P1", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W7-1/8", "W8", "W9", "W10", "X1")
FRAME_SIZES = (4, 16, 64)


def flat_splits(by_split: dict[str, SirenParams]) -> dict[str, torch.Tensor]:
    return {s: p.flat() for s, p in by_split.items()}


def build_rung(name: str, cache: CorpusCache, dataset: str, device: str) -> Rung:
    """Feature maps of docs/prereg/S1.md §1."""
    if name == "P0":
        feats, labels = cache.images("P-shared-det")
        return Rung("P0", feats, labels, notes="real pixels")

    if name == "P1":
        by_split, labels = cache.split_params("P-shared-det")
        feats = {s: render_features(p, dataset, device) for s, p in by_split.items()}
        return Rung("P1", feats, labels, notes="oracle render of the fitted INR")

    if name in ("W1", "W2", "W3"):
        protocol = {"W1": "P-shared-det", "W2": "P-shared-stoch", "W3": "P-random"}[name]
        by_split, labels = cache.split_params(protocol)
        return Rung(name, flat_splits(by_split), labels, notes=f"raw weights, {protocol}")

    if name in ("W4", "W5"):
        by_split, labels = cache.split_params("P-random")
        if name == "W4":
            fn = lambda p: c_sort(p)[0].flat()  # noqa: E731
            note = "c_sort canonicalization"
        else:
            template = shared_init_template(by_split["train"])
            probes = probe_coords(dataset)
            fn = lambda p: c_align(p, template, probes)[0].flat()  # noqa: E731
            note = "c_align canonicalization against the shared init theta_0"
        feats = {s: _chunked(fn, p) for s, p in by_split.items()}
        return Rung(name, feats, labels, notes=note)

    if name == "W6":
        by_split, labels = cache.split_params("P-random")
        return Rung(
            "W6", flat_splits(by_split), labels,
            params_train=by_split["train"], augment=bounded_aug,
            flatten=lambda p: p.flat(),
            notes="raw weights + bounded group augmentation at train time",
        )

    if name in ("W7", "W7-1/8"):
        by_split, labels = cache.split_params("P-random")
        k_split, k_labels = cache.split_params("P-random-K")
        feats = {s: by_split[s].flat() for s in SPLITS}
        lab = {s: labels[s] for s in SPLITS}
        if name == "W7":
            feats["train"], lab["train"] = k_split["train"].flat(), k_labels["train"]
            note = "K-marginalization: all K=8 views as training rows"
        else:
            keep = torch.arange(0, k_split["train"].batch, 8)  # one view per image
            feats["train"] = _index(k_split["train"], keep).flat()
            lab["train"] = k_labels["train"][keep]
            note = "W7 control: K corpus subsampled to one view per image (row-count matched)"
        return Rung(name, feats, lab, notes=note)

    if name == "W8":
        by_split, labels = cache.split_params("P-random")
        canon = {s: c_sort(p)[0] for s, p in by_split.items()}
        return Rung(
            "W8", flat_splits(canon), labels,
            params_train=canon["train"], augment=bounded_aug,
            flatten=lambda p: p.flat(),
            notes="c_sort canonicalization + bounded augmentation in the canonical frame",
        )

    if name == "W9":
        by_split, labels = cache.split_params("P-random")
        frames = {r: build_frame(by_split["train"], seed=0, size=r) for r in FRAME_SIZES}
        best = max(FRAME_SIZES)  # cells for every R are reported; the decoder runs on the largest
        feats = {s: frame_average(p, frames[best]) for s, p in by_split.items()}
        return Rung("W9", feats, labels, notes=f"frame averaging, R={best}, frame fixed per seed")

    if name == "W10":
        by_split, labels = cache.split_params("P-random")
        feats = {s: _chunked(encode_deep, p) for s, p in by_split.items()}
        return Rung("W10", feats, labels, notes="deep phase-invariant encoding (L=2, Ch3.6)")

    raise ValueError(f"unknown rung {name}")


def run_rung(rung: Rung, seeds: int, device: str, extra_eval=None) -> dict:
    accs, extra_accs, epochs = [], [], []
    t0 = time.time()
    for s in range(seeds):
        res = train_matched_mlp(
            rung.feats, rung.labels, seed=s, device=device,
            augment=rung.augment, params_train=rung.params_train, flatten=rung.flatten,
            extra_eval=extra_eval,
        )
        accs.append(res.test_acc)
        epochs.append(res.epochs_ran)
        if res.extra_acc:
            extra_accs.append(res.extra_acc)
        print(f"  {rung.name} seed {s}: test {res.test_acc:.2f} "
              f"(val {res.val_acc:.2f}, {res.epochs_ran} ep)", flush=True)
    cell = {
        "rung": rung.name,
        "notes": rung.notes,
        "feature_dim": int(rung.feats["train"].shape[1]),
        "n_train_rows": int(rung.feats["train"].shape[0]),
        "seeds": seeds,
        "augmentation_bearing": rung.is_augmentation_bearing,
        "acc": accs,
        "mean": float(np.mean(accs)),
        "ci95_bootstrap": bootstrap_ci_mean(np.array(accs)),
        "epochs_ran": epochs,
        "wallclock_s": time.time() - t0,
    }
    if extra_accs:
        cell["extra_acc"] = {
            k: [d[k] for d in extra_accs] for k in extra_accs[0]
        }
    if not rung.is_augmentation_bearing:
        cell["linear_probe"] = linear_probe(rung.feats, rung.labels, seed=0, device=device)
        cell["knn10_cosine"] = knn_accuracy(rung.feats, rung.labels)
    else:
        cell["linear_probe"] = None  # identical to the raw rung by construction (prereg §3)
        cell["knn10_cosine"] = None
    return cell


def label_shuffle_control(rung: Rung, device: str, seed: int = 0) -> float:
    """Retrain on permuted training labels; must collapse to chance (prereg §5)."""
    gen = torch.Generator().manual_seed(seed)
    shuffled = dict(rung.labels)
    perm = torch.randperm(len(shuffled["train"]), generator=gen)
    shuffled["train"] = shuffled["train"][perm]
    res = train_matched_mlp(rung.feats, shuffled, seed=seed, device=device)
    return res.test_acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--rungs", nargs="+", default=["all"])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--out", default="results/ladder")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shuffle-controls", nargs="*", default=["W1", "W3", "W5"])
    args = ap.parse_args()

    wanted = list(ALL_RUNGS) if args.rungs == ["all"] else args.rungs
    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    out_dir = Path(args.out) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in wanted:
        dest = out_dir / f"{name.replace('/', '-')}.json"
        if dest.exists() and not args.force:
            print(f"{name}: cell exists, skipping", flush=True)
            continue
        print(f"=== rung {name} ===", flush=True)
        if name == "X1":
            w1 = build_rung("W1", cache, args.dataset, args.device)
            w3 = build_rung("W3", cache, args.dataset, args.device)
            cell = run_rung(
                Rung("X1", w1.feats, w1.labels, notes="train on W1, evaluate on W3 features"),
                SEEDS_DETERMINISTIC, args.device,
                extra_eval={"W3_test": (w3.feats["test"], w3.labels["test"])},
            )
            reverse = run_rung(
                Rung("X1-rev", w3.feats, w3.labels, notes="train on W3, evaluate on W1 features"),
                SEEDS_DETERMINISTIC, args.device,
                extra_eval={"W1_test": (w1.feats["test"], w1.labels["test"])},
            )
            cell["reverse"] = reverse
        else:
            rung = build_rung(name, cache, args.dataset, args.device)
            seeds = SEEDS_AUGMENTED if rung.is_augmentation_bearing else SEEDS_DETERMINISTIC
            cell = run_rung(rung, seeds, args.device)
            if name in args.shuffle_controls:
                cell["label_shuffle_test_acc"] = label_shuffle_control(rung, args.device)
                print(f"  {name} label-shuffle control: "
                      f"{cell['label_shuffle_test_acc']:.2f}", flush=True)
        dest.write_text(json.dumps(cell, indent=2))
        print(f"  -> {dest} ({cell['mean']:.2f} mean over {cell['seeds']} seeds)", flush=True)


if __name__ == "__main__":
    main()

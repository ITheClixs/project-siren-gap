#!/usr/bin/env python3
"""INR-Bench generation (Ch4): fit a dataset under one protocol, shard-resumable.

Usage:
  .venv/bin/python scripts/03_generate_inrbench.py --dataset mnist \
      --protocol P-shared-det --split all --steps 1000 --width 32 --layers 2 \
      [--limit N] [--batch 256] [--device mps] [--out-root data/inrbench] [--tag pilot]

Subsetting (CIFAR fallback path, docs/COMPUTE_LEDGER.md): --n-train/--n-val/--n-test
take a prefix of each split's id list, e.g. --n-train 20000 --n-val 2000 --n-test 2000.
--limit truncates the concatenated id list instead (pilot use).

Resume: shards named shard_<split>_<start>.safetensors (+ .parquet); existing pairs
are skipped, so restarts continue where they left off (checkpoint interval = one
shard ≈ 20 s at B=256 on MPS, far under the 10-minute protocol requirement).

Seed policy (docs/THINKING/G3-design.md):
  P-shared-det    init_seed 0 (all), full-batch, fit_seed 0
  P-shared-stoch  init_seed 0 (all), coord_batch 256, fit_seed 500000+shard_start
  P-random        init_seed 1000000+image_id (train/val), 3000000+image_id (test), full-batch
  P-random-K      train only, K=8: init_seed 10000000+image_id*10+k
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sirengap.data.images import DATASET_SPECS, load_dataset, spec_of  # noqa: E402
from sirengap.data.schema import save_shard, validate_metadata  # noqa: E402
from sirengap.fitting.batched import fit_batch, make_coord_grid, psnr  # noqa: E402

PROTOCOLS = ("P-shared-det", "P-shared-stoch", "P-random", "P-random-K")
TEST_ID_OFFSET = 100000


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def split_of(image_id: int, val_start: int) -> str:
    if image_id >= TEST_ID_OFFSET:
        return "test"
    return "val" if image_id >= val_start else "train"


def init_seed_for(protocol: str, image_id: int, k: int = 0) -> int:
    if protocol in ("P-shared-det", "P-shared-stoch"):
        return 0
    if protocol == "P-random":
        return (3000000 if image_id >= TEST_ID_OFFSET else 1000000) + image_id
    if protocol == "P-random-K":
        return 10000000 + image_id * 10 + k
    raise ValueError(protocol)


def generate(args: argparse.Namespace) -> None:
    device = args.device
    spec = spec_of(args.dataset)
    coords = make_coord_grid(spec.side, spec.side)
    widths = tuple([args.width] * args.layers)

    x_train, y_train = load_dataset(args.dataset, "train")
    x_test, y_test = load_dataset(args.dataset, "test")
    if len(x_train) != spec.n_train_file:
        raise ValueError(f"{args.dataset} train file has {len(x_train)} rows, spec says {spec.n_train_file}")

    def images_for(ids: list[int]) -> torch.Tensor:
        rows = [x_test[i - TEST_ID_OFFSET] if i >= TEST_ID_OFFSET else x_train[i] for i in ids]
        return torch.stack(rows)

    def label_of(i: int) -> int:
        return int(y_test[i - TEST_ID_OFFSET] if i >= TEST_ID_OFFSET else y_train[i])

    wanted_splits = {"all": ("train", "val", "test"), "train": ("train",), "val": ("val",), "test": ("test",)}[args.split]

    def prefix(all_ids: list[int], n: int) -> list[int]:
        return all_ids[:n] if n else all_ids

    ids: list[int] = []
    if "train" in wanted_splits:
        ids += prefix(list(range(0, spec.val_start)), args.n_train)
    if "val" in wanted_splits:
        ids += prefix(list(range(spec.val_start, spec.n_train_file)), args.n_val)
    if "test" in wanted_splits:
        ids += prefix([TEST_ID_OFFSET + i for i in range(len(x_test))], args.n_test)
    if args.limit:
        ids = ids[: args.limit]
    ks = list(range(8)) if args.protocol == "P-random-K" else [0]

    out = Path(args.out_root) / args.dataset / (args.protocol + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": args.dataset, "protocol": args.protocol, "steps": args.steps, "lr": args.lr,
        "width": args.width, "layers": args.layers, "batch": args.batch, "code_version": git_hash(),
        "torch": torch.__version__, "device": device, "limit": args.limit, "split": args.split,
        "side": spec.side, "channels": spec.channels, "val_start": spec.val_start,
        "n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test, "n_ids": len(ids),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    total_done = 0
    t_start = time.time()
    for k in ks:
        for start in range(0, len(ids), args.batch):
            chunk = ids[start : start + args.batch]
            stem = f"shard_k{k}_{start:06d}" if len(ks) > 1 else f"shard_{start:06d}"
            spath, mpath = out / f"{stem}.safetensors", out / f"{stem}.parquet"
            if spath.exists() and mpath.exists():
                total_done += len(chunk)
                continue
            t0 = time.time()
            targets = images_for(chunk)
            iseeds = [init_seed_for(args.protocol, i, k) for i in chunk]
            kwargs = dict(widths=widths, steps=args.steps, lr=args.lr, device=device,
                          init_seeds=iseeds)
            if args.protocol == "P-shared-stoch":
                kwargs.update(coord_batch=256, fit_seed=500000 + start)
            result = fit_batch(targets, coords, **kwargs)
            wall = time.time() - t0
            save_shard(result.params, spath)
            rows = pd.DataFrame(
                {
                    "image_id": chunk,
                    "label": [label_of(i) for i in chunk],
                    "split": [split_of(i, spec.val_start) for i in chunk],
                    "activation": "sine",
                    "protocol": args.protocol,
                    "init_seed": iseeds,
                    "fit_seed": kwargs.get("fit_seed", 0),
                    "steps": args.steps,
                    "lr": args.lr,
                    "final_psnr": psnr(result.final_loss).numpy(),
                    "final_loss": result.final_loss.numpy(),
                    # relative endpoint gradient norm: the stationarity measure S8 needs,
                    # since render fidelity does not distinguish "interpolates the grid"
                    # from "sits at a stationary point"
                    "final_grad_norm": (
                        result.final_grad_norm.numpy()
                        if result.final_grad_norm is not None
                        else float("nan")
                    ),
                    "wallclock_s": wall / len(chunk),
                    "code_version": config["code_version"],
                }
            )
            rows.to_parquet(mpath)
            total_done += len(chunk)
            rate = total_done / max(time.time() - t_start, 1e-9)
            print(f"{stem}: {len(chunk)} fits in {wall:.1f}s  psnr_med={float(psnr(result.final_loss).median()):.1f}dB  overall {rate:.1f} fits/s", flush=True)

    # merge + validate
    parts = sorted(out.glob("shard_*.parquet"))
    meta = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    validate_metadata(meta)
    meta.to_parquet(out / "metadata.parquet")
    corr_psnr = float(meta["final_psnr"].corr(meta["label"]))
    summary = {
        "n_inrs": len(meta), "psnr_median": float(meta["final_psnr"].median()),
        "psnr_q05": float(meta["final_psnr"].quantile(0.05)),
        "corr_psnr_label": corr_psnr, "wallclock_total_s": time.time() - t_start,
        "grad_norm_median": float(meta["final_grad_norm"].median()),
        "grad_norm_q95": float(meta["final_grad_norm"].quantile(0.95)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=sorted(DATASET_SPECS))
    ap.add_argument("--protocol", required=True, choices=PROTOCOLS)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=0, help="0 = all train ids")
    ap.add_argument("--n-val", type=int, default=0, help="0 = all val ids")
    ap.add_argument("--n-test", type=int, default=0, help="0 = all test ids")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out-root", default="data/inrbench")
    ap.add_argument("--tag", default="")
    generate(ap.parse_args())


if __name__ == "__main__":
    main()

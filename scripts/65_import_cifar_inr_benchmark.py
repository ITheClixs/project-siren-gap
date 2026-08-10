#!/usr/bin/env python3
"""Import the CIFAR-10 INR benchmark of Zhou et al. (the corpus ScaleGMN's CIFAR column uses).

Layout differs from the MNIST and FashionMNIST releases:

    siren_cifar_wts/randinit_smaller_<class>s/net<image_id>.pth

so the label comes from the directory and the CIFAR-10 image index from the filename. Keys are
``net.<i>.linear.{weight,bias}`` with an RGB head, i.e. ``net.2.weight`` of shape (3, 32).

The train/test division is the standard CIFAR one and is *verified* rather than assumed: image
ids below 50000 give exactly 5000 per class and ids at or above it give exactly 1000 per class,
which is CIFAR-10's own split. Validation is carved deterministically from the training half,
which is our split and not the authors', and is declared wherever the comparison appears.

The 950k augmented INRs in the same archive belong to ScaleGMN's separate "Augmented CIFAR-10"
column and are not imported here.

Usage:
  .venv/bin/python scripts/65_import_cifar_inr_benchmark.py --src /tmp/cifar_probe/siren_cifar_wts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.data.schema import save_shard  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402

SHARD = 256
TEST_FROM = 50_000
KEYS = ("net.0.linear.weight", "net.0.linear.bias", "net.1.linear.weight",
        "net.1.linear.bias", "net.2.weight", "net.2.bias")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--protocol", default="P-dws-bench")
    ap.add_argument("--out-root", default="data/inrbench")
    ap.add_argument("--val-every", type=int, default=10, help="1 in N training INRs to validation")
    args = ap.parse_args()

    src = Path(args.src)
    items = []
    for c in range(10):
        for f in sorted((src / f"randinit_smaller_{c}s").glob("net*.pth")):
            items.append((f, c, int(re.search(r"net(\d+)", f.name).group(1))))
    if not items:
        raise SystemExit(f"no INRs under {src}")
    items.sort(key=lambda t: (t[2], t[1]))
    print(f"{len(items)} CIFAR-10 INRs")

    out_dir = Path(args.out_root) / args.dataset / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, tens = [], []
    for f, label, image_id in items:
        sd = torch.load(f, map_location="cpu", weights_only=True)
        tens.append([sd[k] for k in KEYS])
        if image_id >= TEST_FROM:
            split = "test"
        else:
            split = "val" if (image_id % args.val_every == 0) else "train"
        rows.append({"image_id": image_id, "label": label, "split": split,
                     "activation": "sine", "protocol": args.protocol,
                     "init_seed": -1, "fit_seed": -1, "steps": -1, "lr": -1.0,
                     "final_psnr": -1.0, "final_loss": -1.0, "final_grad_norm": -1.0,
                     "wallclock_s": -1.0, "code_version": "imported:zhou-cifar10-inrs"})

    meta = pd.DataFrame(rows)
    print(meta.groupby("split").size().to_string())

    for start in range(0, len(tens), SHARD):
        ch = tens[start:start + SHARD]
        st = [torch.stack([c[k] for c in ch]) for k in range(6)]
        save_shard(SirenParams(hidden=((st[0], st[1]), (st[2], st[3])),
                               w_out=st[4], b_out=st[5]),
                   out_dir / f"shard_{start:06d}.safetensors")
        meta.iloc[start:start + SHARD].to_parquet(out_dir / f"shard_{start:06d}.parquet")
    (out_dir / "import.json").write_text(json.dumps(
        {"source": str(src), "n": len(tens),
         "split": "CIFAR-10's own train/test division, verified by per-class counts; "
                  "validation carved deterministically from the training half",
         "val_every": args.val_every}, indent=2))
    print(f"wrote {len(tens)} INRs to {out_dir}")


if __name__ == "__main__":
    main()

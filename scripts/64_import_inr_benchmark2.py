#!/usr/bin/env python3
"""Import benchmark INR corpora distributed in the ``splits.json`` layout.

The FashionMNIST-INR and CIFAR10-INR releases used by DWSNets / NFN / ScaleGMN ship as

    <root>/{train,test}/model_<k>.pth      each holding the state dict plus an int ``label``
    <root>/splits.json                     {"train": [...], "val": [...], "test": [...]}

which is a better import than the MNIST archive: the split is the authors' own rather than one we
carve, and the labels come from the checkpoints rather than from directory names. Paths inside
splits.json are absolute from the authors' machine, so entries are matched by basename.

Usage:
  .venv/bin/python scripts/64_import_inr_benchmark2.py \
      --src /tmp/fmnist_probe/fmnist_inrs --dataset fashionmnist --protocol P-dws-bench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.data.schema import save_shard  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402

SHARD = 256
KEYS = ("seq.0.weight", "seq.0.bias", "seq.1.weight", "seq.1.bias",
        "seq.2.weight", "seq.2.bias")


def split_map(root: Path) -> dict[str, str]:
    spl = json.loads((root / "splits.json").read_text())
    out: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for p in spl[name]:
            # keyed by "<dir>/<file>", because the train and test halves number their
            # checkpoints independently and so collide on basename alone
            q = Path(p)
            out[f"{q.parent.name}/{q.name}"] = name
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", default="P-dws-bench")
    ap.add_argument("--out-root", default="data/inrbench")
    args = ap.parse_args()

    src = Path(args.src)
    smap = split_map(src)
    files = sorted([p for d in ("train", "test") for p in (src / d).glob("model_*.pth")],
                   key=lambda p: (p.parent.name, int(p.stem.split("_")[1])))
    print(f"{len(files)} checkpoints, {len(smap)} split entries")

    out_dir = Path(args.out_root) / args.dataset / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, tens, missing = [], [], 0
    for f in files:
        sd = torch.load(f, map_location="cpu", weights_only=True)
        split = smap.get(f"{f.parent.name}/{f.name}")
        if split is None:                      # present on disk but not in the authors' split
            missing += 1
            continue
        tens.append([sd[k] for k in KEYS])
        # ids are namespaced by the directory the file came from, because the two halves
        # restart their numbering independently
        uid = int(f.stem.split("_")[1]) + (10_000_000 if f.parent.name == "test" else 0)
        rows.append({"image_id": uid, "label": int(sd["label"]), "split": split,
                     "activation": "sine", "protocol": args.protocol,
                     "init_seed": -1, "fit_seed": -1, "steps": -1, "lr": -1.0,
                     "final_psnr": -1.0, "final_loss": -1.0, "final_grad_norm": -1.0,
                     "wallclock_s": -1.0, "code_version": f"imported:{args.dataset}-inrs"})

    meta = pd.DataFrame(rows)
    print(meta.groupby("split").size().to_string())
    if missing:
        print(f"skipped {missing} checkpoints absent from splits.json")

    for start in range(0, len(tens), SHARD):
        ch = tens[start:start + SHARD]
        st = [torch.stack([c[k] for c in ch]) for k in range(6)]
        save_shard(SirenParams(hidden=((st[0], st[1]), (st[2], st[3])),
                               w_out=st[4], b_out=st[5]),
                   out_dir / f"shard_{start:06d}.safetensors")
        meta.iloc[start:start + SHARD].to_parquet(out_dir / f"shard_{start:06d}.parquet")
    (out_dir / "import.json").write_text(json.dumps(
        {"source": str(src), "n": len(tens), "splits": "authors' own splits.json",
         "labels": "from the checkpoints"}, indent=2))
    print(f"wrote {len(tens)} INRs to {out_dir}")


if __name__ == "__main__":
    main()

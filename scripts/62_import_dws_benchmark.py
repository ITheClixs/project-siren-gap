#!/usr/bin/env python3
"""Import the standard MNIST-INR benchmark (Navon et al., used by DWSNets/NFN/GMN/ScaleGMN).

This is the corpus the published weight-space leaderboard is measured on. Its INRs have the
same architecture as ours -- 2 hidden layers of width 32, 2-D input, 1-D output, sine -- so our
readers can be run on it without adaptation, which turns "our reader beats our readers" into a
comparison against published numbers.

Two properties measured on import rather than assumed:

* the corpus is *independently initialized*. Median pairwise relative parameter distance over a
  random sample is 1.42, against 1.40 for our own P-random and 0.19 for our shared-init corpus.
  So the published leaderboard is measured in the hard regime, not the easy one.
* labels and the train/test division come from the directory names
  (``mnist_png_training_5_51886`` -> split ``training``, label 5). The official
  ``mnist_splits.json`` is not in the distributed archive; we therefore carve validation out of
  the training half by a deterministic rule and record that this is our split, not theirs.

Usage:
  .venv/bin/python scripts/62_import_dws_benchmark.py --src /tmp/mnist_inrs_probe/mnist-inrs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.data.schema import save_shard  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
NAME_RE = re.compile(r"mnist_png_(training|testing)_(\d)_(\d+)$")
SHARD = 256


def collect(src: Path) -> list[tuple[Path, str, int, int]]:
    out = []
    for d in sorted(src.iterdir()):
        m = NAME_RE.match(d.name)
        if not m:
            continue
        ck = d / "checkpoints" / "model_final.pth"
        if ck.exists():
            out.append((ck, m.group(1), int(m.group(2)), int(m.group(3))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-root", default="data/inrbench")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--protocol", default="P-dws-bench")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    items = collect(Path(args.src))
    if not items:
        raise SystemExit(f"no INRs found under {args.src}")
    print(f"found {len(items)} MNIST INRs")

    # deterministic validation carve-out from the training half, by image id
    train_ids = sorted({i for _, s, _, i in items if s == "training"})
    n_val = int(round(args.val_frac * len(train_ids)))
    val_ids = set(train_ids[::max(len(train_ids) // n_val, 1)][:n_val])

    out_dir = Path(args.out_root) / args.dataset / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, tensors = [], []
    for ck, half, label, image_id in items:
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        tensors.append([sd["seq.0.weight"], sd["seq.0.bias"], sd["seq.1.weight"],
                        sd["seq.1.bias"], sd["seq.2.weight"], sd["seq.2.bias"]])
        split = "test" if half == "testing" else ("val" if image_id in val_ids else "train")
        # The archive restarts its numbering between the training and testing halves, so the two
        # halves collide in image_id while referring to different images. Namespacing the test
        # half keeps the id space unique; the schema's split-disjointness check catches this.
        uid = image_id + (10_000_000 if half == "testing" else 0)
        rows.append({"image_id": uid, "label": label, "split": split,
                     "activation": "sine", "protocol": args.protocol,
                     # sentinels, not measurements: this corpus was fitted by its authors and
                     # the archive carries no fit metadata. -1 marks "not applicable here".
                     "init_seed": -1, "fit_seed": -1, "steps": -1, "lr": -1.0,
                     "final_psnr": -1.0, "final_loss": -1.0,
                     "final_grad_norm": -1.0, "wallclock_s": -1.0,
                     "code_version": "imported:navon-dwsnets-mnist-inrs"})

    meta = pd.DataFrame(rows)
    print(meta.groupby("split").size().to_string())

    for start in range(0, len(tensors), SHARD):
        chunk = tensors[start:start + SHARD]
        st = [torch.stack([c[k] for c in chunk]) for k in range(6)]
        save_shard(SirenParams(hidden=((st[0], st[1]), (st[2], st[3])),
                               w_out=st[4], b_out=st[5]),
                   out_dir / f"shard_{start:06d}.safetensors")
        meta.iloc[start:start + SHARD].to_parquet(out_dir / f"shard_{start:06d}.parquet")
    (out_dir / "import.json").write_text(json.dumps(
        {"source": str(args.src), "n": len(items),
         "note": "standard MNIST-INR benchmark; validation carved deterministically from the "
                 "training half because the official splits file is not in the archive",
         "val_frac": args.val_frac}, indent=2))
    print(f"wrote {len(tensors)} INRs to {out_dir}")


if __name__ == "__main__":
    main()

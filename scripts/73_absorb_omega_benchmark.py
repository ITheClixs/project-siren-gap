#!/usr/bin/env python3
"""Bring an imported INR benchmark corpus into the canonical stored form sin(Wx + b).

The published INR-classification archives store SIREN checkpoints in the internal
parameterization, where every hidden layer computes sin(omega_0 * (W h + b)) with omega_0 = 30.
Our own fitter stores the canonical form, with omega_0 absorbed into W and b
(`fitting.batched.absorb_omega`), and every group action, invariance audit and phasor feature in
this repository assumes that form. Scripts 62, 64 and 65 imported the archives verbatim, so the
stored first-layer biases sit within about 0.1 of zero and the phasor coordinates degenerate.

This script writes a sibling corpus `<protocol>-w0` with each hidden (W, b) multiplied by
omega_0 and the output layer untouched, which realizes exactly the function the archive's own
forward pass computes. Nothing in the source corpus is modified.

    .venv/bin/python scripts/73_absorb_omega_benchmark.py --dataset mnist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.constants import OMEGA_0  # noqa: E402
from sirengap.data.schema import load_shard, save_shard, validate_metadata  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402


def absorb(params: SirenParams, omega: float) -> SirenParams:
    hidden = tuple((w * omega, b * omega) for w, b in params.hidden)
    return SirenParams(hidden=hidden, w_out=params.w_out, b_out=params.b_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", default="P-dws-bench")
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--omega", type=float, default=float(OMEGA_0))
    args = ap.parse_args()

    src = ROOT / args.root / args.dataset / args.protocol
    dst_protocol = f"{args.protocol}-w0"
    dst = ROOT / args.root / args.dataset / dst_protocol
    shards = sorted(src.glob("shard_*.safetensors"))
    if not shards:
        raise SystemExit(f"no shards in {src}")
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} already exists; remove it deliberately before regenerating")
    dst.mkdir(parents=True)

    metas = []
    for shard in shards:
        save_shard(absorb(load_shard(shard), args.omega), dst / shard.name)
        meta = pd.read_parquet(shard.with_suffix(".parquet"))
        meta["protocol"] = dst_protocol
        meta["code_version"] = meta["code_version"].astype(str) + "+omega-absorbed"
        meta.to_parquet(dst / shard.with_suffix(".parquet").name, index=False)
        metas.append(meta)
    validate_metadata(pd.concat(metas, ignore_index=True))

    manifest = {
        "derived_from": str(src.relative_to(ROOT)),
        "transform": f"hidden (W, b) multiplied by omega_0 = {args.omega}; output layer unchanged",
        "reason": "archive stores the internal SIREN parameterization; the repository's group "
                  "actions and phasor features assume the canonical form sin(Wx + b)",
        "n_shards": len(shards),
    }
    (dst / "import.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(shards)} shards to {dst}")


if __name__ == "__main__":
    main()

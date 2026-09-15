#!/usr/bin/env python3
"""S19 arm A9: export in-house corpora in the storage format of the published INR benchmarks.

docs/prereg/S19.md. NG-GNN (neural-graphs) and ScaleGMN read one state dict per INR, with keys
seq.{0,1,2}.{weight,bias}, and a splits JSON of paths and labels. As registered, the weights are
written in the canonical form the corpora are stored in (omega_0 absorbed into W and b), not divided
back by omega_0. That matters for ScaleGMN: its phase canonicalization folds the *stored* bias, so
only in this form does the fold act on the phase each network actually computes. The readers take
weights as data and never evaluate the network. The split assignment is the in-house one.
ScaleGMN resolves paths relative to its repository and rewrites the second path component for its
canonicalized copy, so a second splits file with repository-relative paths is written for it.

    .venv/bin/python scripts/84_export_dws_format.py --protocol P-random --out ../external/data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.canon.csort import _phase_reduce  # noqa: E402
from sirengap.data.schema import load_corpus  # noqa: E402
from sirengap.models.params import outgoing, replace_layer  # noqa: E402

SPLITS = ("train", "val", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--protocol", default="P-random")
    ap.add_argument("--out", required=True, help="parent directory for the exported corpus")
    ap.add_argument("--fold", action="store_true",
                    help="fold every hidden bias into [-pi/2, pi/2) first (ScaleGMN's Algorithm 1 in "
                         "canonical form; function-preserving)")
    args = ap.parse_args()

    params, meta = load_corpus(ROOT / "data" / "inrbench" / args.dataset / args.protocol)
    if args.fold:
        for layer in range(params.n_layers):
            w, b = params.hidden[layer]
            w, b, out_w = _phase_reduce(w, b, outgoing(params, layer))
            params = replace_layer(params, layer, w, b, out_w)
    tag = "-fold" if args.fold else ""
    dest = (Path(args.out) / f"sirengap-{args.dataset}-{args.protocol}{tag}").resolve()
    (dest / "inrs").mkdir(parents=True, exist_ok=True)
    splits: dict = {s: {"path": [], "label": []} for s in SPLITS}
    for i in range(params.batch):
        sd = OrderedDict()
        for layer, (w, b) in enumerate(params.hidden):
            sd[f"seq.{layer}.weight"] = w[i].clone()
            sd[f"seq.{layer}.bias"] = b[i].clone()
        n = len(params.hidden)
        sd[f"seq.{n}.weight"] = params.w_out[i].clone()
        sd[f"seq.{n}.bias"] = params.b_out[i].clone()
        path = dest / "inrs" / f"inr_{i:06d}.pth"
        torch.save(sd, path)
        row = meta.iloc[i]
        splits[row["split"]]["path"].append(str(path))
        splits[row["split"]]["label"].append(int(row["label"]))
    (dest / "splits.json").write_text(json.dumps(splits))
    rel = {s: {"path": [f"data/{dest.name}/inrs/{Path(q).name}" for q in v["path"]],
               "label": v["label"]} for s, v in splits.items()}
    (dest / "splits_rel.json").write_text(json.dumps(rel))
    (dest / "export.json").write_text(json.dumps({
        "source": f"data/inrbench/{args.dataset}/{args.protocol}", "n": params.batch,
        "transform": ("hidden biases folded into [-pi/2, pi/2) by g_{0,j}; otherwise " if args.fold else "")
        + "canonical form as stored (omega_0 absorbed into hidden W and b)",
        "counts": {s: len(v["label"]) for s, v in splits.items()}}, indent=2))
    print(f"wrote {params.batch} INRs to {dest}", {s: len(v["label"]) for s, v in splits.items()})


if __name__ == "__main__":
    main()

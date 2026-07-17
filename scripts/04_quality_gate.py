#!/usr/bin/env python3
"""Quality-gate + audits for a generated INR corpus (protocol Ch4).

Trains/caches the pixel CNN, renders the corpus shards, and reports:
task-referenced gate (renders within 1.0 pt of real-pixel accuracy),
corr(PSNR, label) / corr(loss, label) leakage audit, strata audit (PO-3).

Usage:
  .venv/bin/python scripts/04_quality_gate.py --dir data/inrbench/mnist/P-shared-det \
      --dataset mnist --eval-split val [--device mps] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sirengap.data.images import DATA_DIR, load_idx_dataset  # noqa: E402
from sirengap.data.schema import load_shard  # noqa: E402
from sirengap.eval.quality_gate import accuracy, quality_gate, train_gate_cnn  # noqa: E402
from sirengap.fitting.batched import make_coord_grid  # noqa: E402
from sirengap.geometry.audit import strata_audit  # noqa: E402
from sirengap.models.forward import forward_canonical  # noqa: E402

TEST_ID_OFFSET = 100000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.dir)
    meta = pd.read_parquet(corpus / "metadata.parquet")
    sel = meta[meta["split"] == args.eval_split]
    if args.limit:
        sel = sel.iloc[: args.limit]
    if sel.empty:
        raise SystemExit(f"no rows for split {args.eval_split} in {corpus}")

    x_train, y_train = load_idx_dataset(args.dataset, "train")
    x_test, y_test = load_idx_dataset(args.dataset, "test")
    model = train_gate_cnn(
        x_train[:55000], y_train[:55000], side=28, device=args.device,
        cache=DATA_DIR / f"gate_cnn_{args.dataset}.pt",
    )

    # render selected INRs shard by shard
    coords = make_coord_grid(28, 28, device=args.device)
    wanted = set(sel["image_id"].tolist())
    renders, labels, reals, audits = [], [], [], []
    for spath in sorted(corpus.glob("shard_*.safetensors")):
        mpath = spath.with_suffix(".parquet")
        rows = pd.read_parquet(mpath)
        mask = rows["image_id"].isin(wanted).to_numpy()
        if not mask.any():
            continue
        params = load_shard(spath).to(args.device)
        with torch.no_grad():
            out = forward_canonical(params, coords).clamp(-1, 1).cpu()
        idx = torch.from_numpy(mask.nonzero()[0])
        renders.append(out[idx])
        audits.append(strata_audit(load_shard(spath)))
        for i in rows.loc[mask, "image_id"]:
            labels.append(int(y_test[i - TEST_ID_OFFSET] if i >= TEST_ID_OFFSET else y_train[i]))
            reals.append(x_test[i - TEST_ID_OFFSET] if i >= TEST_ID_OFFSET else x_train[i])
    renders_t = torch.cat(renders)
    reals_t = torch.stack(reals)
    labels_t = torch.tensor(labels)

    gate = quality_gate(model, reals_t, renders_t, labels_t, args.device)
    report = {
        "corpus": str(corpus), "eval_split": args.eval_split, "n_eval": len(labels_t),
        "gate": gate,
        "leakage": {
            "corr_psnr_label": float(sel["final_psnr"].corr(sel["label"])),
            "corr_loss_label": float(sel["final_loss"].corr(sel["label"])),
        },
        "psnr": {"median": float(sel["final_psnr"].median()), "q05": float(sel["final_psnr"].quantile(0.05))},
        "strata_audit_first_shard": audits[0] if audits else None,
        "gate_cnn_train_acc_check": accuracy(model, x_train[:2000], y_train[:2000], args.device),
    }
    dest = Path("results/inrbench")
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}_{corpus.name}_{args.eval_split}_gate.json"
    (dest / name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

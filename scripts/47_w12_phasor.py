#!/usr/bin/env python3
"""Rung W12 — the phasor-graded reader, on raw parameters (docs/prereg/S9.md).

W11a is permutation-equivariant and blind to D_infinity. W11b is G-invariant but only because
it is handed W10's fixed invariant family. W12 is the third thing: a reader that quotients
D_infinity on the **raw parameters**, by putting the bias in phasor coordinates so the infinite
winding collapses to a parity, and then keeping the resulting Z_2 x Z_2 grading through every
layer. T16 asserts the logits are invariant to windings up to |j| = 40.

Capacity is set by rule, not by choice: the width minimising |params - 1,873,162|, the frozen
decoder's parameter count, exactly as S1-w11.md fixed for W11a/W11b.

Usage:
  .venv/bin/python scripts/47_w12_phasor.py --dataset mnist
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.eval.rungs import SPLITS, CorpusCache  # noqa: E402
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
from sirengap.models.phasor import (  # noqa: E402
    CHARACTERS,
    PhasorGradedReader,
    apply_scale,
    feature_scale,
    phasor_features,
)
from sirengap.models.readers import count_parameters  # noqa: E402

SEEDS = 5
MAX_EPOCHS = 100
PATIENCE = 10
BATCH = 512
LR = 1e-3
DECODER_PARAMS = 1_873_162  # the frozen matched MLP


def matched_width(feats: dict, graded: bool = True, lo: int = 32, hi: int = 320) -> int:
    """The capacity rule: minimise |params - DECODER_PARAMS| over widths.

    Applied separately to the graded reader and to its ungraded control, so the two are matched
    in capacity rather than in width -- the control mixes blocks and so needs a smaller width to
    reach the same parameter count.
    """
    best, best_gap = lo, float("inf")
    for w in range(lo, hi + 1):
        n = count_parameters(PhasorGradedReader.from_features(feats, width=w, graded=graded))
        if abs(n - DECODER_PARAMS) < best_gap:
            best, best_gap = w, abs(n - DECODER_PARAMS)
    return best


def extract(by_split: dict[str, SirenParams], chunk: int = 4096) -> dict[str, dict]:
    """Graded features per split, in chunks so a 60k-INR corpus fits in memory."""
    out: dict[str, dict] = {}
    for split in SPLITS:
        p = by_split[split]
        l1: dict = {c: [] for c in CHARACTERS}
        l2: dict = {c: [] for c in CHARACTERS}
        edge = []
        for i in range(0, p.batch, chunk):
            idx = torch.arange(i, min(i + chunk, p.batch))
            sub = SirenParams(
                hidden=tuple((w[idx], b[idx]) for w, b in p.hidden),
                w_out=p.w_out[idx], b_out=p.b_out[idx],
            )
            f = phasor_features(sub)
            for c in CHARACTERS:
                l1[c].append(f["l1"][c])
                l2[c].append(f["l2"][c])
            edge.append(f["edge"])
        out[split] = {
            "l1": {c: torch.cat(l1[c], 0) for c in CHARACTERS},
            "l2": {c: torch.cat(l2[c], 0) for c in CHARACTERS},
            "edge": torch.cat(edge, 0),
        }
    return out


def slice_feats(feats: dict, idx: torch.Tensor, device: str) -> dict:
    return {
        "l1": {c: feats["l1"][c][idx].to(device) for c in CHARACTERS},
        "l2": {c: feats["l2"][c][idx].to(device) for c in CHARACTERS},
        "edge": feats["edge"][idx].to(device),
    }


@torch.no_grad()
def accuracy(model: nn.Module, feats: dict, y: torch.Tensor, device: str,
             batch: int = 1024) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(y), batch):
        idx = torch.arange(i, min(i + batch, len(y)))
        correct += int((model(slice_feats(feats, idx, device)).argmax(1).cpu() == y[idx]).sum())
    return 100.0 * correct / len(y)


def train_one(fs: dict, labels: dict, seed: int, device: str, width: int,
              max_epochs: int = MAX_EPOCHS, graded: bool = True) -> dict:
    torch.manual_seed(seed)
    model = PhasorGradedReader.from_features(fs["train"], width=width, graded=graded).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    y_tr = labels["train"]

    best_val, best_state, best_epoch = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(y_tr), generator=gen)
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(slice_feats(fs["train"], idx, device)), y_tr[idx].to(device)).backward()
            opt.step()
        sched.step()
        val = accuracy(model, fs["val"], labels["val"], device)
        if val > best_val:
            best_val, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return {
        "test_acc": accuracy(model, fs["test"], labels["test"], device),
        "val_acc": best_val,
        "epochs_ran": best_epoch + 1,
        "params": count_parameters(model),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--protocol", default="P-random")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--width", type=int, default=0, help="0 = the capacity rule")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--out-name", default="")
    ap.add_argument("--ungraded", action="store_true",
                    help="the matched control: same skeleton and capacity, grading removed")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, labels = cache.split_params(args.protocol)
    ladder = ROOT / "results" / "ladder" / args.dataset
    anchors = {r: json.loads((ladder / f"{r}.json").read_text())["acc"] for r in ("W1", "W3")}
    w1, w3 = np.array(anchors["W1"]), np.array(anchors["W3"])

    t0 = time.time()
    feats = extract(by_split)
    stats = feature_scale(feats["train"])
    fs = {s: apply_scale(feats[s], stats) for s in SPLITS}
    del feats

    graded = not args.ungraded
    width = args.width or matched_width(fs["train"], graded=graded)
    print(f"width {width} (capacity rule against the decoder's {DECODER_PARAMS:,})", flush=True)

    accs, epochs, params_n = [], [], 0
    for s in range(args.seeds):
        r = train_one(fs, labels, s, args.device, width, args.max_epochs, graded=graded)
        accs.append(r["test_acc"])
        epochs.append(r["epochs_ran"])
        params_n = r["params"]
        print(f"  {'W12' if graded else 'W12u'} seed {s}: test {r['test_acc']:.2f} (val {r['val_acc']:.2f}, "
              f"{r['epochs_ran']} ep)", flush=True)

    a = np.array(accs)
    n = min(len(a), len(w1), len(w3))
    f = (a[:n] - w3[:n]) / (w1[:n] - w3[:n])
    out = {
        "dataset": args.dataset, "protocol": args.protocol, "prereg": "docs/prereg/S9.md",
        "W1": float(w1.mean()), "W3": float(w3.mean()),
        "width": width, "reader_params": params_n, "graded": graded,
        "acc": accs, "mean": float(a.mean()), "ci95_bootstrap": bootstrap_ci_mean(a),
        "recovery_fraction": float(f.mean()), "f_ci95": bootstrap_ci_mean(f),
        "epochs_ran": epochs, "wallclock_s": time.time() - t0,
    }
    print(f"{name}: {out['mean']:.2f}  f={out['recovery_fraction']:.4f}  "
          f"params={params_n:,}  ({out['wallclock_s']:.0f}s)", flush=True)

    name = args.out_name or ("W12" if graded else "W12u")
    path = ladder / f"{name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rung W11 — equivariant weight-space readers, against the ladder's frozen frames.

Every other rung changes the feature map and holds the reader fixed at the matched MLP. W11
does the opposite: same corpus (`P-random`), same splits, same seeds, same optimiser protocol,
but the reader is permutation-equivariant. Two variants:

  W11a  RawGraphReader        bipartite message passing on raw weights. S_n-equivariant only,
                              which is the coverage the DWSNets/NFN/GMN family has for sine
                              networks. Answers: does reader architecture substitute for frame
                              choice?
  W11b  InvariantGraphReader  W10's D_infinity-invariants with *learned* equivariant pooling in
                              place of sorted eigenvalue spectra. Answers OPEN_PROBLEMS #4's
                              remaining half: is the pooling the bottleneck, or the invariants?

Because W11 deliberately breaks the matched-reader control, parameter count and wallclock are
reported next to accuracy, and the registration fixes a capacity band rather than leaving it free.

Usage:
  .venv/bin/python scripts/33_w11_equivariant.py --dataset mnist --variants a b
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
from sirengap.models.readers import (  # noqa: E402
    InvariantGraphReader,
    RawGraphReader,
    apply_stats,
    count_parameters,
    feature_stats,
    invariant_graph_features,
    raw_graph_features,
)

SEEDS = 5
MAX_EPOCHS = 100
PATIENCE = 10
BATCH = 512
LR = 1e-3


def extract(params_by_split, variant: str, chunk: int = 4096) -> dict[str, dict[str, torch.Tensor]]:
    """Feature dicts per split, built in chunks so a 60k-INR corpus fits in memory."""
    fn = raw_graph_features if variant == "a" else invariant_graph_features
    out: dict[str, dict[str, torch.Tensor]] = {}
    for split in SPLITS:
        p = params_by_split[split]
        pieces: dict[str, list[torch.Tensor]] = {}
        for i in range(0, p.batch, chunk):
            idx = torch.arange(i, min(i + chunk, p.batch))
            sub = type(p)(
                hidden=tuple((w[idx], b[idx]) for w, b in p.hidden),
                w_out=p.w_out[idx], b_out=p.b_out[idx],
            )
            for k, v in fn(sub).items():
                pieces.setdefault(k, []).append(v)
        out[split] = {k: torch.cat(v, dim=0) for k, v in pieces.items()}
    return out


@torch.no_grad()
def accuracy(model: nn.Module, feats: dict[str, torch.Tensor], y: torch.Tensor,
             device: str, batch: int = 1024) -> float:
    model.eval()
    correct = 0
    n = len(y)
    for i in range(0, n, batch):
        sl = {k: v[i : i + batch].to(device) for k, v in feats.items()}
        correct += int((model(sl).argmax(1).cpu() == y[i : i + batch]).sum())
    return 100.0 * correct / n


def train_one(feats, labels, variant: str, seed: int, device: str, width: int,
              max_epochs: int = MAX_EPOCHS) -> dict:
    """Same schedule as the frozen matched MLP: AdamW 1e-3, cosine, <=100 epochs, patience 10."""
    torch.manual_seed(seed)
    stats = feature_stats(feats["train"])
    fs = {s: apply_stats(feats[s], stats) for s in SPLITS}

    if variant == "a":
        model = RawGraphReader(
            m=fs["train"]["x1"].shape[2] - 1, c=fs["train"]["x2"].shape[2] - 1, width=width
        ).to(device)
    else:
        model = InvariantGraphReader(
            n_node=fs["train"]["x1"].shape[2], n_edge=fs["train"]["e"].shape[3],
            n_global=fs["train"]["g"].shape[1], width=width,
        ).to(device)

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
            xb = {k: v[idx].to(device) for k, v in fs["train"].items()}
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), y_tr[idx].to(device)).backward()
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
    ap.add_argument("--variants", nargs="+", default=["a", "b"])
    # Capacity-matched to the frozen decoder (1,873,162 params) to within ~1%, so that a loss
    # against the MLP rungs cannot be blamed on the reader being smaller. Rule, not free choice.
    ap.add_argument("--width-a", type=int, default=424)
    ap.add_argument("--width-b", type=int, default=288)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--protocol", default="P-random",
                    help="S6 arm (iii) needs the unscattered P-shared-det baseline for H-S6-5")
    ap.add_argument("--out-name", default="W11",
                    help="cell name under results/ladder/<dataset>; never overwrite W11.json "
                         "with a non-default protocol")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, labels = cache.split_params(args.protocol)
    ladder = ROOT / "results" / "ladder" / args.dataset
    anchors = {r: json.loads((ladder / f"{r}.json").read_text())["acc"] for r in ("W1", "W3")}
    w1, w3 = np.array(anchors["W1"]), np.array(anchors["W3"])

    out = {"dataset": args.dataset, "prereg": "docs/prereg/S1-w11.md",
           "protocol": args.protocol,
           "W1": float(w1.mean()), "W3": float(w3.mean()), "variants": {}}

    for variant in args.variants:
        t0 = time.time()
        feats = extract(by_split, variant)
        accs, params_n, epochs = [], 0, []
        for s in range(args.seeds):
            width = args.width_a if variant == "a" else args.width_b
            r = train_one(feats, labels, variant, s, args.device, width, args.max_epochs)
            accs.append(r["test_acc"])
            epochs.append(r["epochs_ran"])
            params_n = r["params"]
            print(f"  W11{variant} seed {s}: test {r['test_acc']:.2f} "
                  f"(val {r['val_acc']:.2f}, {r['epochs_ran']} ep)", flush=True)
        a = np.array(accs)
        n = min(len(a), len(w1), len(w3))
        f = (a[:n] - w3[:n]) / (w1[:n] - w3[:n])
        out["variants"][f"W11{variant}"] = {
            "width": args.width_a if variant == "a" else args.width_b,
            "acc": accs, "mean": float(a.mean()), "ci95_bootstrap": bootstrap_ci_mean(a),
            "recovery_fraction": float(f.mean()), "f_ci95": bootstrap_ci_mean(f),
            "reader_params": params_n, "epochs_ran": epochs,
            "wallclock_s": time.time() - t0,
        }
        r = out["variants"][f"W11{variant}"]
        print(f"W11{variant}: {r['mean']:.2f}  f={r['recovery_fraction']:.4f}  "
              f"params={params_n:,}  ({r['wallclock_s']:.0f}s)", flush=True)
        del feats

    if args.protocol != "P-random" and args.out_name == "W11":
        raise SystemExit("refusing to overwrite W11.json with a non-default protocol; "
                         "pass --out-name")
    path = ladder / f"{args.out_name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

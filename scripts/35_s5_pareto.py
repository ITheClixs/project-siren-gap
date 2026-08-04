#!/usr/bin/env python3
"""S5 — the FLOPs-matched adjudication of weight access against function access (RQ6).

Proposition PO-6 says a *complete* invariant of the weights carries exactly the information of the
realised function. If so, weight access cannot win on accuracy, and the field's justification must
be computational. S5 builds the frontier that settles it.

Three access models on the same corpora:

  function-query   learned probe coordinates, classify the queried outputs. Provably nuisance-free
                   (T14), so its accuracy should not depend on the fitting protocol at all — which
                   this script checks directly rather than assuming.
  render           the same thing at every grid point (rung P1's information, priced).
  weight           the ladder's rungs, whose accuracies are already measured.

FLOPs come from the analytic accounting in `sirengap.eval.flops`, not from wall-clock, so the
frontier is reproducible on any machine. Both regimes PO-6's corollary distinguishes are reported:
single-task, and amortized over T downstream tasks where the weight-space preprocessing is paid
once but function-query must re-learn its probes per task.

Usage:
  .venv/bin/python scripts/35_s5_pareto.py --dataset mnist --probes 1 4 16 64 256
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

from sirengap.data.images import spec_of  # noqa: E402
from sirengap.eval.flops import (  # noqa: E402
    Arch,
    function_query,
    weight_calign,
    weight_csort,
    weight_equivariant_reader,
    weight_phasor_reader,
    weight_invariants,
    weight_raw,
)
from sirengap.eval.probes import ProbeReader  # noqa: E402
from sirengap.eval.rungs import SPLITS, CorpusCache  # noqa: E402
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402

MAX_EPOCHS = 100
SEEDS = 5  # frozen in docs/prereg/S5.md section 6 for the headline K sweep
PATIENCE = 10
BATCH = 512
LR = 1e-3


def index_params(p: SirenParams, idx: torch.Tensor) -> SirenParams:
    return SirenParams(
        hidden=tuple((w[idx], b[idx]) for w, b in p.hidden),
        w_out=p.w_out[idx], b_out=p.b_out[idx],
    )


def to_device(p: SirenParams, device: str) -> SirenParams:
    return p.to(device)


@torch.no_grad()
def accuracy(model: nn.Module, params: SirenParams, y: torch.Tensor,
             device: str, batch: int = 1024) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(y), batch):
        idx = torch.arange(i, min(i + batch, len(y)))
        pred = model(index_params(params, idx)).argmax(1).cpu()
        correct += int((pred == y[idx]).sum())
    return 100.0 * correct / len(y)


def train_probe_reader(by_split, labels, n_probes: int, seed: int, device: str,
                       out_dim: int, freeze: bool = False, max_epochs: int = MAX_EPOCHS) -> dict:
    """Same schedule as the frozen decoder; only the probes and the head are trained."""
    torch.manual_seed(seed)
    p_tr = to_device(by_split["train"], device)
    p_va = to_device(by_split["val"], device)
    p_te = to_device(by_split["test"], device)
    model = ProbeReader(n_probes=n_probes, out_dim=out_dim, freeze_probes=freeze).to(device)

    with torch.no_grad():
        sample = index_params(p_tr, torch.arange(min(4096, p_tr.batch)))
        model.set_normalization(model.features(sample))

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
            loss_fn(model(index_params(p_tr, idx.to(device))), y_tr[idx].to(device)).backward()
            opt.step()
        sched.step()
        val = accuracy(model, p_va, labels["val"], device)
        if val > best_val:
            best_val, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return {
        "test_acc": accuracy(model, p_te, labels["test"], device),
        "val_acc": best_val,
        "epochs_ran": best_epoch + 1,
        "probe_drift": float((model.probes.detach().cpu() - ProbeReader(
            n_probes=n_probes, out_dim=out_dim).probes.detach()).norm()) if not freeze else 0.0,
    }


def weight_points(dataset: str, arch: Arch) -> list[dict]:
    """Weight-access rungs: accuracies already measured, priced by the analytic accounting."""
    ladder = ROOT / "results" / "ladder" / dataset
    ana = json.loads((ladder / "S1_analysis.json").read_text())
    m = ana["means"]
    pts = [
        ("W3 raw weights", m.get("W3"), weight_raw(arch)),
        ("W4 c_sort", m.get("W4"), weight_csort(arch)),
        ("W5 c_align", m.get("W5"), weight_calign(arch, 256)),
        ("W10 invariants", m.get("W10"), weight_invariants(arch, 320)),
        ("W1 shared-init (ceiling)", m.get("W1"), weight_raw(arch)),
    ]
    w11 = ladder / "W11.json"
    if w11.exists():
        v = json.loads(w11.read_text())["variants"]
        pts.append(("W11a equivariant (raw)", v["W11a"]["mean"],
                    weight_equivariant_reader(arch, v["W11a"]["width"],
                                              invariant_features=False, n_global=0)))
        pts.append(("W11b equivariant (invariant)", v["W11b"]["mean"],
                    weight_equivariant_reader(arch, v["W11b"]["width"])))

    w12_path = ROOT / "results" / "ladder" / dataset / "W12.json"
    if w12_path.exists():
        w12 = json.loads(w12_path.read_text())
        pts.append(("W12 phasor-graded", w12["mean"],
                    weight_phasor_reader(arch, w12["width"])))
    return [
        {"name": n, "access": "weight", "acc": a, "flops": c["per_inr"],
         "amortized": c.get("amortized", 0), "preprocess": c.get("preprocess", 0)}
        for n, a, c in pts if a is not None
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--probes", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--protocol", default="P-random")
    ap.add_argument("--nuisance-control", action="store_true",
                    help="repeat one probe count on P-shared-det: function access must not care")
    ap.add_argument("--control-probes", type=int, default=16)
    ap.add_argument("--frozen-ablation", action="store_true",
                    help="also run fixed (unlearned) probes, to separate querying from learning")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--exploratory", action="store_true",
                    help="permit an off-protocol seed count; writes a separate _EXPLORATORY "
                         "artifact that the scorer refuses to read")
    args = ap.parse_args()

    # The same mechanical guard 48_s8_sweep.py carries, extended to this path after a chain step
    # re-ran the sweep at n=3 and silently overwrote the registered n=5 artifact (CLAIMS row 54).
    if not args.exploratory and args.seeds != SEEDS:
        raise SystemExit(
            f"refusing to overwrite the registered artifact off-protocol (seeds={args.seeds}); "
            f"docs/prereg/S5.md section 6 fixes n={SEEDS} for the headline K sweep, and permits 3 "
            "only for K=256 and only if the sweep exceeds 3 h. Use --exploratory to write a "
            "separate artifact that is not scoreable."
        )

    spec = spec_of(args.dataset)
    arch = Arch(in_dim=2, width=32, layers=2, out_dim=spec.channels)
    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, labels = cache.split_params(args.protocol)

    out: dict = {
        "dataset": args.dataset, "protocol": args.protocol,
        "prereg": "docs/prereg/S5.md",
        "arch": {"in_dim": 2, "width": 32, "layers": 2, "out_dim": spec.channels,
                 "n_params": arch.n_params},
        "function_query": [], "weight_access": weight_points(args.dataset, arch),
        "controls": {},
    }

    for k in args.probes:
        t0 = time.time()
        accs = [
            train_probe_reader(by_split, labels, k, s, args.device, spec.channels,
                               max_epochs=args.max_epochs)["test_acc"]
            for s in range(args.seeds)
        ]
        cost = function_query(arch, k, n_classes=10)
        a = np.array(accs)
        out["function_query"].append({
            "name": f"function-query K={k}", "access": "function", "n_probes": k,
            "acc": float(a.mean()), "acc_ci95": bootstrap_ci_mean(a), "seeds": accs,
            "flops": cost["per_inr"], "probe_eval_flops": cost["probe_eval"],
            "wallclock_s": time.time() - t0,
        })
        r = out["function_query"][-1]
        print(f"function-query K={k:4d}: {r['acc']:6.2f}  {r['flops']/1e6:7.3f} MFLOP/INR "
              f"({r['wallclock_s']:.0f}s)", flush=True)

    if args.nuisance_control:
        k = args.control_probes
        other = "P-shared-det" if args.protocol == "P-random" else "P-random"
        bs2, lab2 = cache.split_params(other)
        acc = train_probe_reader(bs2, lab2, k, 0, args.device, spec.channels,
                                 max_epochs=args.max_epochs)["test_acc"]
        base = next(r for r in out["function_query"] if r["n_probes"] == k)["seeds"][0]
        out["controls"]["nuisance_invariance"] = {
            "probes": k, "protocol_a": args.protocol, "acc_a": base,
            "protocol_b": other, "acc_b": acc, "difference": acc - base,
            "note": "function access is provably protocol-invariant (T14); this measures it",
        }
        print(f"nuisance control K={k}: {args.protocol} {base:.2f} vs {other} {acc:.2f} "
              f"(diff {acc - base:+.2f})", flush=True)

    if args.frozen_ablation:
        for k in (args.control_probes,):
            acc = train_probe_reader(by_split, labels, k, 0, args.device, spec.channels,
                                     freeze=True, max_epochs=args.max_epochs)["test_acc"]
            base = next(r for r in out["function_query"] if r["n_probes"] == k)["seeds"][0]
            out["controls"][f"frozen_probes_K{k}"] = {
                "learned": base, "frozen": acc, "gain_from_learning": base - acc,
            }
            print(f"frozen-probe ablation K={k}: learned {base:.2f} vs fixed {acc:.2f} "
                  f"(learning buys {base - acc:+.2f})", flush=True)

    name = f"pareto_{args.dataset}.json" if not args.exploratory \
        else f"pareto_{args.dataset}_EXPLORATORY.json"
    path = ROOT / "results" / "s5" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    if args.exploratory:
        print("exploratory run: this artifact is off-protocol and 36_score_s5.py will not read it")


if __name__ == "__main__":
    main()

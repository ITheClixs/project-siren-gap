#!/usr/bin/env python3
"""How far does a fit travel from its initialization? (PO-9 laziness, at corpus scale)

The microcosm says the fit stays in its initialization's neighbourhood, which is why a
shared initialization is a shared frame and why aligning to a fixed reference works at all.
This measures the same quantity on the real corpora, per dataset, with no new fitting:

    relative travel   r = ||theta_T - theta_0|| / ||theta_0||          (P-shared-det only,
                                                                       where theta_0 is known)
    per-neuron cosine c = cos angle between w_i(T) and w_i(0), layer 1

If the alignment rung's recovery fraction tracks travel across datasets, the drop in
f(W5) on CIFAR-10 is a statement about how far its fits ran, not about natural images.

Usage:
  .venv/bin/python scripts/23_fit_travel.py --datasets mnist fashionmnist cifar10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.eval.rungs import CorpusCache, shared_init_template  # noqa: E402


def travel_stats(dataset: str, root: Path, n_max: int) -> dict:
    cache = CorpusCache(root / dataset, dataset)
    params, _ = cache.corpus("P-shared-det")
    n = min(n_max, params.batch)
    theta0 = shared_init_template(params)

    flat = params.flat()[:n].double()
    flat0 = theta0.flat().double()                       # [1, D]
    delta = (flat - flat0).norm(dim=1)
    rel = delta / flat0.norm()

    # layer-1 per-neuron direction change
    w1 = params.hidden[0][0][:n].double()                # [n, n1, m]
    w1_0 = theta0.hidden[0][0].double()                  # [1, n1, m]
    cos = torch.nn.functional.cosine_similarity(w1, w1_0.expand_as(w1), dim=2)  # [n, n1]

    # bias winding: how many multiples of pi the phase moved
    b1 = params.hidden[0][1][:n].double()
    b1_0 = theta0.hidden[0][1].double()
    wind = ((b1 - b1_0).abs() / torch.pi)

    q = lambda t, p: float(torch.quantile(t.flatten(), p))  # noqa: E731
    return {
        "dataset": dataset,
        "n_inrs": int(n),
        "steps": int(json.loads((root / dataset / "P-shared-det" / "config.json").read_text())["steps"]),
        "rel_travel": {"median": q(rel, 0.5), "q05": q(rel, 0.05), "q95": q(rel, 0.95)},
        "abs_travel": {"median": q(delta, 0.5)},
        "init_norm": float(flat0.norm()),
        "layer1_dir_cosine_to_init": {
            "median": q(cos, 0.5), "q05": q(cos, 0.05), "q95": q(cos, 0.95),
            "frac_below_0.9": float((cos < 0.9).double().mean()),
        },
        "layer1_bias_shift_in_pi": {"median": q(wind, 0.5), "q95": q(wind, 0.95)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashionmnist", "cifar10"])
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--n-max", type=int, default=20000)
    ap.add_argument("--out", default="results/fit_travel.json")
    args = ap.parse_args()

    out = {"note": "PO-9 laziness measured on P-shared-det corpora; no new fitting", "by_dataset": {}}
    for ds in args.datasets:
        if not (Path(args.root) / ds / "P-shared-det").exists():
            print(f"{ds}: skipped (no corpus)")
            continue
        s = travel_stats(ds, Path(args.root), args.n_max)
        out["by_dataset"][ds] = s
        print(
            f"{ds:14s} steps={s['steps']:5d}  rel_travel={s['rel_travel']['median']:.4f}  "
            f"layer1 cos-to-init={s['layer1_dir_cosine_to_init']['median']:.4f}  "
            f"frac<0.9={s['layer1_dir_cosine_to_init']['frac_below_0.9']:.3f}  "
            f"|db|/pi={s['layer1_bias_shift_in_pi']['median']:.4f}",
            flush=True,
        )

    # pair with the alignment recovery fraction where the ladder has landed
    for ds, s in out["by_dataset"].items():
        ana = ROOT / "results" / "ladder" / ds / "S1_analysis.json"
        if ana.exists():
            d = json.loads(ana.read_text())
            s["f_W5"] = d["recovery_fractions"].get("f_W5", {}).get("point")
            s["f_W10"] = d["recovery_fractions"].get("f_W10", {}).get("point")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

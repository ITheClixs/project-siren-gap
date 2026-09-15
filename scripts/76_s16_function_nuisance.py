#!/usr/bin/env python3
"""S16: function access on P-random against P-shared-det, five paired seeds per probe count.

docs/prereg/S16.md. Replaces the single-seed nuisance control of scripts/35_s5_pareto.py.

    .venv/bin/python scripts/76_s16_function_nuisance.py --dataset mnist
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sirengap.data.images import spec_of  # noqa: E402
from sirengap.eval.rungs import CorpusCache, probe_coords  # noqa: E402
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402

S5 = __import__("35_s5_pareto")

SEEDS = 5
PROTOCOLS = ("P-random", "P-shared-det")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--probes", nargs="+", type=int, default=[16, 64, 256])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--max-epochs", type=int, default=S5.MAX_EPOCHS)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--grid-probes", action="store_true",
                    help="S19 A4: freeze the probes at K pixel centres of the fit grid")
    args = ap.parse_args()

    channels = spec_of(args.dataset).channels
    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    data = {p: cache.split_params(p) for p in PROTOCOLS}
    out: dict = {"study": "S16 function-side nuisance control", "prereg": "docs/prereg/S16.md",
                 "dataset": args.dataset, "seeds": args.seeds, "by_probes": {}}

    for k in args.probes:
        t0 = time.time()
        init = probe_coords(args.dataset, k) if args.grid_probes else None
        accs = {p: [S5.train_probe_reader(*data[p], k, s, args.device, channels,
                                          max_epochs=args.max_epochs, freeze=args.grid_probes,
                                          probe_init=init)["test_acc"]
                    for s in range(args.seeds)] for p in PROTOCOLS}
        diff = np.array(accs["P-shared-det"]) - np.array(accs["P-random"])
        out["by_probes"][str(k)] = {
            p: {"acc": a, "mean": float(np.mean(a)), "ci95": bootstrap_ci_mean(np.array(a))}
            for p, a in accs.items()}
        out["by_probes"][str(k)]["difference"] = {
            "paired": diff.tolist(), "mean": float(diff.mean()), "ci95": bootstrap_ci_mean(diff)}
        out["by_probes"][str(k)]["wallclock_s"] = time.time() - t0
        print(f"K={k:4d}: shared-det {np.mean(accs['P-shared-det']):6.2f}  random "
              f"{np.mean(accs['P-random']):6.2f}  D = {diff.mean():+.2f}", flush=True)

    if args.grid_probes:
        out.update({"study": "S19 A4 grid-fixed probes", "prereg": "docs/prereg/S19.md"})
    sub = ("s19", "_grid") if args.grid_probes else ("s16", "")
    path = ROOT / "results" / sub[0] / f"function_nuisance_{args.dataset}{sub[1]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

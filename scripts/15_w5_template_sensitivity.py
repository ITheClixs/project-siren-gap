#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): how much of W5 depends on *which* template c_align uses?

W5 recovered f = 0.628 of the W1-W3 gap by aligning P-random INRs to the corpus's shared
initialization theta_0. theta_0 is available only because the corpus was built with a known
shared init, so the practical reading of that number hinges on whether alignment to an
*arbitrary* template does as well. This runs the same rung against several templates.

Watermarked exploratory per protocol: results go to results/ladder/<dataset>/EXPLORATORY_*.json
and may not be reported as confirmatory evidence.

Usage: .venv/bin/python scripts/15_w5_template_sensitivity.py [--dataset mnist] [--seeds 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sirengap.canon.calign import c_align  # noqa: E402
from sirengap.eval.decoder import linear_probe, train_matched_mlp  # noqa: E402
from sirengap.eval.rungs import CorpusCache, _chunked, _index, probe_coords  # noqa: E402
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402
from sirengap.fitting.batched import absorb_omega, init_from_seeds  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402


def template_from_seed(like: SirenParams, seed: int) -> SirenParams:
    return absorb_omega(
        init_from_seeds([seed], like.hidden[0][0].shape[2], like.widths(), like.w_out.shape[1])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    args = ap.parse_args()

    cache = CorpusCache(Path(args.root) / args.dataset, args.dataset)
    by_split, labels = cache.split_params("P-random")
    probes = probe_coords(args.dataset)

    w1_split, _ = cache.split_params("P-shared-det")
    templates = {
        "theta0_shared_init": template_from_seed(by_split["train"], 0),  # the frozen W5 choice
        "unrelated_init_12345": template_from_seed(by_split["train"], 12345),
        "unrelated_init_777": template_from_seed(by_split["train"], 777),
        "a_fitted_shared_det_inr": _index(w1_split["train"], torch.tensor([0])),
        "a_fitted_random_inr": _index(by_split["train"], torch.tensor([0])),
    }

    out: dict[str, dict] = {}
    for name, template in templates.items():
        feats = {
            s: _chunked(lambda p: c_align(p, template, probes)[0].flat(), q)
            for s, q in by_split.items()
        }
        accs = [
            train_matched_mlp(feats, labels, seed=s, device=args.device).test_acc
            for s in range(args.seeds)
        ]
        out[name] = {
            "acc": accs,
            "mean": float(np.mean(accs)),
            "ci95_bootstrap": bootstrap_ci_mean(np.array(accs)),
            "linear_probe": linear_probe(feats, labels, seed=0, device=args.device),
        }
        print(f"{name}: {np.mean(accs):.2f} (probe {out[name]['linear_probe']:.2f})", flush=True)

    ladder = Path("results/ladder") / args.dataset
    cells = {p.stem: json.loads(p.read_text()) for p in ladder.glob("W[13].json")}
    w1, w3 = float(np.mean(cells["W1"]["acc"])), float(np.mean(cells["W3"]["acc"]))
    for name, cell in out.items():
        cell["recovery_fraction"] = (cell["mean"] - w3) / (w1 - w3)

    report = {
        "status": "EXPLORATORY — not pre-registered, not confirmatory evidence",
        "question": "does W5's recovery depend on the template being the shared init?",
        "W1": w1, "W3": w3, "seeds": args.seeds, "templates": out,
    }
    dest = ladder / "EXPLORATORY_w5_template_sensitivity.json"
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: round(v["recovery_fraction"], 3) for k, v in out.items()}, indent=2))
    print(f"written {dest}")


if __name__ == "__main__":
    main()

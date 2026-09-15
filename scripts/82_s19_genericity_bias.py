#!/usr/bin/env python3
"""S19 arms A7 and A8: genericity on the stratum the theorems exclude, and the benchmark bias range.

docs/prereg/S19.md. The strata audit measured the smallest angle between the lines spanned by two
first-layer rows, which is the parallel stratum, not the one Theorem L=1 excludes (w_j = +-w_i as
vectors). A7 measures delta = min_{i != j} min(|w_i - w_j|, |w_i + w_j|) / median_i |w_i| on fitted
P-random INRs and on fresh draws from the same initialization, so the fitted value can be read
against what initialization alone gives. A8 reports, for each published benchmark corpus after
omega_0 absorption, the fraction of hidden biases outside the fold interval [-pi/2, pi/2).

    .venv/bin/python scripts/82_s19_genericity_bias.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.data.schema import load_corpus  # noqa: E402
from sirengap.fitting.batched import absorb_omega, init_from_seeds  # noqa: E402

N_INIT = 10000
THRESHOLDS = (1e-3, 1e-2, 1e-1)
DATASETS = ("mnist", "fashionmnist", "cifar10")


def delta(w: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """w [B, n, m] -> [B]: nearest pair up to sign, relative to the median row norm."""
    outs = []
    for i in range(0, w.shape[0], chunk):
        x = w[i:i + chunk].double()
        diff = (x[:, :, None, :] - x[:, None, :, :]).norm(dim=-1)
        summ = (x[:, :, None, :] + x[:, None, :, :]).norm(dim=-1)
        d = torch.minimum(diff, summ)
        eye = torch.eye(x.shape[1], dtype=torch.bool)
        d = d.masked_fill(eye[None], float("inf"))
        outs.append(d.amin(dim=(1, 2)) / x.norm(dim=2).median(dim=1).values)
    return torch.cat(outs)


def summary(v: torch.Tensor) -> dict:
    q = torch.quantile(v.float(), torch.tensor([0.01, 0.05, 0.5]))
    return {"n": int(v.numel()), "q01_q05_q50": [float(x) for x in q],
            "frac_below": {str(t): float((v < t).float().mean()) for t in THRESHOLDS}}


def main() -> None:
    out: dict = {"study": "S19 A7 genericity and A8 benchmark bias range",
                 "prereg": "docs/prereg/S19.md"}

    params, _ = load_corpus(ROOT / "data" / "inrbench" / "mnist" / "P-random")
    w_fit = params.hidden[0][0]
    widths = params.widths()
    init = absorb_omega(init_from_seeds(list(range(N_INIT)), w_fit.shape[2], widths,
                                        params.w_out.shape[1]))
    out["A7_delta"] = {"fitted_P-random_mnist": summary(delta(w_fit)),
                       "init_only_mnist": summary(delta(init.hidden[0][0]))}
    print("A7", json.dumps(out["A7_delta"], indent=1), flush=True)

    out["A8_bias_outside_fold"] = {}
    for ds in DATASETS:
        corpus = ROOT / "data" / "inrbench" / ds / "P-dws-bench-w0"
        if not corpus.exists():
            continue
        p, _ = load_corpus(corpus)
        per_layer = {}
        for layer, (_, b) in enumerate(p.hidden):
            outside = (b < -math.pi / 2) | (b >= math.pi / 2)
            per_layer[f"layer{layer}"] = {"frac_outside": float(outside.float().mean()),
                                          "frac_inrs_with_any": float(outside.any(dim=1).float().mean()),
                                          "abs_bias_q50_q95_max": [float(x) for x in torch.quantile(
                                              b.abs().flatten().float()[:2_000_000],
                                              torch.tensor([0.5, 0.95, 1.0]))]}
        out["A8_bias_outside_fold"][ds] = per_layer
        print("A8", ds, per_layer, flush=True)

    path = ROOT / "results" / "s19" / "genericity_bias.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

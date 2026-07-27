#!/usr/bin/env python3
"""PO-8 basin census under the *production* optimizer (G2 advisor Empiricist 1, owed before F4).

`scripts/02_microcosm_po8.py` censused basins with Nelder-Mead on the profiled loss, which is
optimizer-unrealistic in two ways: the corpora are fitted by Adam, and the linear parameters
(u, c) are trained rather than profiled out. A basin structure that only exists under a
derivative-free method on a profiled surface would not license any claim about the corpora.

This script repeats the census on the *full* model u sin(wt+b) + c, fitted on a 64-point grid
of [-1, 1] by Adam and by plain gradient descent, at the production learning rate and step
count as well as a converged setting, and reports the endpoint classes under the same rules
(ridge / global-orbit / spurious) so the two censuses can be read side by side.

Usage: .venv/bin/python scripts/13_microcosm_optimizers.py [--n-inits 900] [--out results/microcosm]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("microcosm", ROOT / "scripts" / "02_microcosm_po8.py")
_micro = importlib.util.module_from_spec(_spec)
sys.modules["microcosm"] = _micro
_spec.loader.exec_module(_micro)

A, OMEGA, PHI, C0 = _micro.A, _micro.OMEGA, _micro.PHI, _micro.C0
N_GRID = 64  # 1-D analogue of the pixel grid the corpora are fitted on
INIT_RANGES = (2.0, 5.0, 10.0, 20.0)
SETTINGS = (
    ("adam_production", "adam", 1e-3, 300),  # exactly the frozen corpus setting
    ("adam_converged", "adam", 1e-2, 5000),
    ("gd_converged", "sgd", 1e-2, 5000),
)


def fit_batch_1d(
    w0: torch.Tensor, b0: torch.Tensor, optimizer: str, lr: float, steps: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit u sin(wt+b) + c to the microcosm target from the given (w, b) inits.

    u and c are *trained*, not profiled out, and start from the SIREN-style uniform draw —
    the point of the exercise is to keep every ingredient the production fitter has.
    """
    t = torch.linspace(-1.0, 1.0, N_GRID, dtype=torch.float64)
    target = A * torch.sin(OMEGA * t + PHI) + C0

    gen = torch.Generator().manual_seed(seed)
    n = len(w0)
    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    u = ((torch.rand(n, generator=gen, dtype=torch.float64) * 2 - 1)).requires_grad_(True)
    c = ((torch.rand(n, generator=gen, dtype=torch.float64) * 2 - 1)).requires_grad_(True)

    params = [w, b, u, c]
    opt = (
        torch.optim.Adam(params, lr=lr)
        if optimizer == "adam"
        else torch.optim.SGD(params, lr=lr)
    )
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = u[:, None] * torch.sin(w[:, None] * t[None, :] + b[:, None]) + c[:, None]
        per_init = ((pred - target[None, :]) ** 2).mean(dim=1)
        per_init.sum().backward()
        opt.step()

    # endpoint gradient norm: a run stopped mid-descent has no basin to be assigned to
    opt.zero_grad(set_to_none=True)
    pred = u[:, None] * torch.sin(w[:, None] * t[None, :] + b[:, None]) + c[:, None]
    final = ((pred - target[None, :]) ** 2).mean(dim=1)
    final.sum().backward()
    grad_norm = torch.stack([p.grad.detach().abs() for p in params]).norm(dim=0)
    return w.detach(), b.detach(), final.detach(), grad_norm


GRAD_TOL = 1e-5  # below this the endpoint is a critical point; above it the run was cut short


def classify(w: float, b: float, loss: float, grad_norm: float, orbit_tol: float = 5e-2) -> str:
    wf, bf = _micro.to_fundamental(w, b)
    if loss < 1e-9 or _micro.orbit_distance(wf, bf) < orbit_tol:
        return "global"
    if grad_norm > GRAD_TOL:
        # still descending: this point is not in any basin, it is simply where the
        # step budget ran out. Counting it as a spurious minimum would be a false claim.
        return "unconverged"
    if _micro.on_degenerate_ridge(wf):
        return "ridge"
    return "spurious"


def census(n_inits: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for label, optimizer, lr, steps in SETTINGS:
        per_range = {}
        for w_range in INIT_RANGES:
            w0 = torch.tensor(rng.uniform(-w_range, w_range, n_inits))
            b0 = torch.tensor(rng.uniform(-np.pi, np.pi, n_inits))
            w_end, b_end, loss, grad = fit_batch_1d(w0, b0, optimizer, lr, steps, seed)
            classes = [
                classify(float(w_end[i]), float(b_end[i]), float(loss[i]), float(grad[i]))
                for i in range(n_inits)
            ]
            moved = (w_end - w0).abs()
            per_range[str(w_range)] = {
                "frac_inits_reaching_global": float(np.mean([c == "global" for c in classes])),
                "frac_inits_on_ridge": float(np.mean([c == "ridge" for c in classes])),
                "frac_inits_spurious": float(np.mean([c == "spurious" for c in classes])),
                "frac_inits_unconverged": float(np.mean([c == "unconverged" for c in classes])),
                "median_abs_w_travel": float(moved.median()),
                "median_final_mse": float(loss.median()),
                "median_endpoint_grad_norm": float(grad.median()),
            }
        out[label] = {"optimizer": optimizer, "lr": lr, "steps": steps, "by_init_range": per_range}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inits", type=int, default=900)
    ap.add_argument("--out", default="results/microcosm")
    args = ap.parse_args()

    report = {
        "n_inits": args.n_inits,
        "n_grid": N_GRID,
        "target": {"A": A, "omega": OMEGA, "phi": PHI, "c0": C0},
        "settings": [
            {"label": s[0], "optimizer": s[1], "lr": s[2], "steps": s[3]} for s in SETTINGS
        ],
        "census": census(args.n_inits),
    }
    nelder = json.loads((ROOT / "results" / "microcosm" / "report.json").read_text())
    report["nelder_mead_reference"] = {
        k: {
            "frac_inits_reaching_global": v["frac_inits_reaching_global"],
            "frac_inits_on_ridge": v["frac_inits_on_ridge"],
            "frac_inits_spurious": v["frac_inits_spurious"],
        }
        for k, v in nelder["basin_census"].items()
    }

    dest = Path(args.out) / "optimizer_census.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["census"], indent=2))
    print(f"\nwritten {dest}")


if __name__ == "__main__":
    main()

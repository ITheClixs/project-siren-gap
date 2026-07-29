#!/usr/bin/env python3
"""S4e — the deep-identifiability falsification hunt (PO-2, L >= 2).

Theorem PO-2 is proved for L = 1: on the generic stratum, f_theta = f_theta' implies
theta' = g theta for a unique g in D_inf wr S_n. For L >= 2 it is Conjecture 6.5, and every
empirical result in this program is L = 2 — the program's weakest theoretical link
(DEFENSE row 15). S4e is the pre-committed empirical attack on it.

The design the proof memo names (PO-2-deep-attempt.md, "Empirical wiring"): fit two-layer
networks to functions realised by *known two-layer teachers*, align, and measure the
recovery rate. Fitting to a teacher — rather than to an image — is what puts the theorem's
actual hypothesis (exact functional equality) within reach.

Four arms:

  planted     theta -> g.theta -> realign. The control that makes the hunt non-vacuous:
              if the search cannot undo an element it knows exists, a large residual on a
              real pair says nothing. Must return machine zero.
  teacher     student fitted to a teacher's exact outputs, width swept. The sharp test.
  null        the same students aligned to an *unrelated* teacher: the scale of "large".
  production  repeated same-image fits from the P-random-K corpus at production width.

The decisive quantity is the joint behaviour of

  R_f  = ||f_student - f_teacher|| / ||f_teacher||     (functional residual, held-out grid)
  R_th = min_{g in G} ||g.theta_s - theta_t|| / ||theta_t||   (orbit residual, scripts/refine)

Identifiable *and* well conditioned  => R_th -> 0 as R_f -> 0, at every width.
Identifiable but ill conditioned     => R_th -> 0 only at small width, degrading with n.
Not identifiable                     => R_th stays large at small width with R_f ~ 0.

Usage:
  .venv/bin/python scripts/26_s4e_identifiability.py --arms planted teacher null
  .venv/bin/python scripts/26_s4e_identifiability.py --arms production --dataset mnist
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

from sirengap.canon.refine import orbit_distance  # noqa: E402
from sirengap.fitting.batched import (  # noqa: E402
    absorb_omega,
    forward_train,
    init_from_seeds,
)
from sirengap.geometry.audit import strata_audit  # noqa: E402
from sirengap.models.forward import forward_canonical  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402
from sirengap.symmetry.dinf import apply, random_element  # noqa: E402

OUT = ROOT / "results" / "s4e"


def dense_grid(side: int, device: str = "cpu") -> torch.Tensor:
    g = torch.linspace(-1.0, 1.0, side)
    return torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1).reshape(-1, 2).to(device)


def teachers(width: int, n: int, seed0: int, layers: int = 2, out_dim: int = 1) -> SirenParams:
    """n independent teachers of the given width; random inits are generically in Theta_gen."""
    return absorb_omega(init_from_seeds(list(range(seed0, seed0 + n)), 2, [width] * layers, out_dim))


def index_batch(p: SirenParams, idx: torch.Tensor) -> SirenParams:
    return SirenParams(
        hidden=tuple((w[idx], b[idx]) for w, b in p.hidden),
        w_out=p.w_out[idx],
        b_out=p.b_out[idx],
    )


def functional_residual(a: SirenParams, b: SirenParams, x: torch.Tensor) -> torch.Tensor:
    """||f_a - f_b|| / ||f_b|| on the probe set, per INR: [B]."""
    with torch.no_grad():
        fa, fb = forward_canonical(a, x), forward_canonical(b, x)
        num = (fa - fb).flatten(1).norm(dim=1)
        den = fb.flatten(1).norm(dim=1).clamp_min(1e-12)
    return num / den


def generic_flags(p: SirenParams) -> dict[str, float]:
    """Fraction of INRs comfortably inside Theta_gen, by the PO-3 strata."""
    a = strata_audit(p)
    out = {}
    for key, vals in a.items():
        v = np.asarray(vals, dtype=float)
        out[f"{key}.min"] = float(v.min())
        out[f"{key}.median"] = float(np.median(v))
    return out


# ------------------------------------------------------------------ arms


def arm_planted(width: int, n: int, device: str, windings: int = 3) -> dict:
    t = teachers(width, n, seed0=0)
    g = random_element(t, torch.Generator().manual_seed(4242), max_windings=windings)
    moved = apply(g, t)
    t0 = time.time()
    rel, diag = orbit_distance(moved, t)
    return {
        "width": width,
        "n": n,
        "R_theta": rel.tolist(),
        "R_theta_median": float(rel.median()),
        "R_theta_max": float(rel.max()),
        "sweeps": diag.sweeps,
        "wallclock_s": time.time() - t0,
    }


def fit_students(
    targets: torch.Tensor,
    coords: torch.Tensor,
    width: int,
    steps: int,
    lr: float,
    init_seeds: list[int],
    device: str,
) -> tuple[SirenParams, list[float], torch.Tensor]:
    """Adam with cosine decay to ~lr/300, full batch.

    `fit_batch` is the corpus fitter and is deliberately frozen at the production schedule
    (constant lr, 300--1000 steps). S4e needs students driven as close to *exact* functional
    equality as the landscape allows, which is a different optimisation problem, so it gets
    its own schedule here rather than a change to the frozen instrument.
    """
    raw = init_from_seeds(init_seeds, coords.shape[1], [width, width], targets.shape[2])
    tensors = [t.to(device).requires_grad_(True) for t in raw]
    tgt, xs = targets.to(device), coords.to(device)
    opt = torch.optim.Adam(tensors, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 300.0)
    curve: list[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        per = ((forward_train(tensors, xs) - tgt) ** 2).mean(dim=(1, 2))
        per.sum().backward()
        opt.step()
        sched.step()
        if step % max(steps // 12, 1) == 0 or step == steps - 1:
            curve.append(float(per.detach().mean().cpu()))
    with torch.no_grad():
        final = ((forward_train(tensors, xs) - tgt) ** 2).mean(dim=(1, 2)).cpu()
    return absorb_omega(tensors), curve, final


def arm_teacher(
    width: int, n: int, steps: int, lr: float, side: int, device: str, seed0: int = 0
) -> dict:
    coords = dense_grid(side)
    t = teachers(width, n, seed0=seed0)
    with torch.no_grad():
        targets = forward_canonical(t, coords)  # [n, P, c] — the teacher's exact function

    t0 = time.time()
    params, curve, final_loss = fit_students(
        targets, coords, width, steps, lr,
        init_seeds=list(range(10_000 + seed0, 10_000 + seed0 + n)), device=device,
    )
    student = params.to("cpu")
    held_out = dense_grid(side + 17)  # off-grid probes: fitting the grid is not fitting f
    r_f = functional_residual(student, t, held_out)
    rel, diag = orbit_distance(student, t)

    return {
        "width": width,
        "n": n,
        "steps": steps,
        "lr": lr,
        "grid_side": side,
        "fit_mse": final_loss.tolist(),
        "R_f": r_f.tolist(),
        "R_theta": rel.tolist(),
        "R_f_min": float(r_f.min()),
        "R_theta_at_best_R_f": float(rel[int(torch.argmin(r_f))]),
        "R_theta_median": float(rel.median()),
        "sweeps": diag.sweeps,
        "student_strata": generic_flags(student),
        "teacher_strata": generic_flags(t),
        "wallclock_s": time.time() - t0,
        "loss_curve": curve,
    }


def _to_internal(p: SirenParams) -> list[torch.Tensor]:
    """Canonical (omega-absorbed) params -> the fitter's internal tensor list."""
    from sirengap.constants import OMEGA_0

    out: list[torch.Tensor] = []
    for w, b in p.hidden:
        out += [w.clone() / OMEGA_0, b.clone() / OMEGA_0]
    return out + [p.w_out.clone(), p.b_out.clone()]


def arm_warmstart(
    width: int, n: int, steps: int, lr: float, side: int, device: str, eps: float = 0.05
) -> dict:
    """Positive control for the *fitting* half: start beside a group image of the teacher.

    `planted` shows the aligner can undo a known element; this shows the optimiser can
    return to the teacher's orbit when it starts within eps of it. Without this control,
    "no independent student recovered" is uninterpretable — it could just mean the fitter
    never recovers anything.
    """
    coords = dense_grid(side)
    t = teachers(width, n, seed0=0)
    with torch.no_grad():
        targets = forward_canonical(t, coords)

    gen = torch.Generator().manual_seed(909)
    g = random_element(t, gen, max_windings=2)
    start = apply(g, t)
    flat = start.flat()
    noise = torch.randn(flat.shape, generator=gen)
    noise = noise / noise.norm(dim=1, keepdim=True) * flat.norm(dim=1, keepdim=True) * eps
    start = _unflatten_like(start, flat + noise)

    tensors = [x.to(device).requires_grad_(True) for x in _to_internal(start)]
    tgt, xs = targets.to(device), coords.to(device)
    opt = torch.optim.Adam(tensors, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 300.0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        ((forward_train(tensors, xs) - tgt) ** 2).mean(dim=(1, 2)).sum().backward()
        opt.step()
        sched.step()
    student = absorb_omega(tensors).to("cpu")

    r_f = functional_residual(student, t, dense_grid(side + 17))
    rel, _ = orbit_distance(student, t)
    return {
        "width": width,
        "n": n,
        "eps_start": eps,
        "R_f_median": float(r_f.median()),
        "R_theta_median": float(rel.median()),
        "R_theta_max": float(rel.max()),
        "R_f_start": float(functional_residual(start, t, dense_grid(side + 17)).median()),
        "R_theta_start": float(orbit_distance(start, t)[0].median()),
        "recovered_frac": float((rel < 1e-3).double().mean()),
    }


def arm_sensitivity(width: int, n: int, side: int, eps_ladder: tuple[float, ...]) -> dict:
    """How ill-conditioned is recovering parameters from a function? No fitting needed.

    Perturb a teacher by eps (relative, in parameter space) and measure both the orbit
    distance and the functional change. Their ratio

        kappa = R_theta / R_f

    is the local condition number of the inverse map f -> theta modulo G. Identifiability
    says the map is injective; kappa says whether that injectivity has any empirical
    content. The proof memo predicts kappa blows up with width (the Bessel--Vandermonde
    determinant it needs collapses from 4.4e-3 at n=2 to 4.4e-13 at n=5), and that
    prediction is testable here directly, at every width, in seconds.
    """
    coords = dense_grid(side)
    t = teachers(width, n, seed0=0)
    gen = torch.Generator().manual_seed(31337)
    rows = []
    for eps in eps_ladder:
        flat = t.flat()
        noise = torch.randn(flat.shape, generator=gen)
        noise = noise / noise.norm(dim=1, keepdim=True) * flat.norm(dim=1, keepdim=True) * eps
        # rebuild a SirenParams with the same shapes from the perturbed flat vector
        pert = _unflatten_like(t, flat + noise)
        r_f = functional_residual(pert, t, coords)
        r_th, _ = orbit_distance(pert, t)
        kappa = r_th / r_f.clamp_min(1e-30)
        rows.append({
            "eps": eps,
            "R_f_median": float(r_f.median()),
            "R_theta_median": float(r_th.median()),
            "kappa_median": float(kappa.median()),
            "kappa_q90": float(torch.quantile(kappa, 0.9)),
        })
    return {"width": width, "n": n, "ladder": rows}


def _unflatten_like(ref: SirenParams, flat: torch.Tensor) -> SirenParams:
    """Inverse of SirenParams.flat() for a batch sharing `ref`'s shapes."""
    hidden, off = [], 0
    for w, b in ref.hidden:
        nw, nb = w[0].numel(), b[0].numel()
        hidden.append((flat[:, off:off + nw].view_as(w), flat[:, off + nw:off + nw + nb].view_as(b)))
        off += nw + nb
    nwo, nbo = ref.w_out[0].numel(), ref.b_out[0].numel()
    w_out = flat[:, off:off + nwo].view_as(ref.w_out)
    b_out = flat[:, off + nwo:off + nwo + nbo].view_as(ref.b_out)
    return SirenParams(hidden=tuple(hidden), w_out=w_out, b_out=b_out)


def arm_null(width: int, n: int, device: str) -> dict:
    """Two unrelated networks of the same shape: the scale of a 'large' residual."""
    a = teachers(width, n, seed0=0)
    b = teachers(width, n, seed0=5000)
    rel, _ = orbit_distance(a, b)
    return {
        "width": width,
        "n": n,
        "R_theta": rel.tolist(),
        "R_theta_median": float(rel.median()),
        "R_theta_min": float(rel.min()),
    }


def arm_production(dataset: str, n_images: int, device: str, root: Path) -> dict:
    """The protocol's literal arm: repeated same-image fits at production L=2, width 32.

    P-random-K holds K = 8 independent fits of each training image. Two fits of one image
    approximate the same target, so their functions are close but not equal; the residual
    after optimal alignment is the production-scale version of the teacher/student number.
    """
    from sirengap.eval.rungs import CorpusCache  # local import: heavy

    cache = CorpusCache(root / dataset, dataset)
    params, meta = cache.corpus("P-random-K")
    k = 8
    n_images = min(n_images, params.batch // k)
    coords = dense_grid(45)

    first = index_batch(params, torch.arange(0, n_images * k, k))
    second = index_batch(params, torch.arange(1, n_images * k + 1, k))

    r_f = functional_residual(first, second, coords)
    rel, diag = orbit_distance(first, second)

    # null within the same corpus: fit of image i against fit of image i+1
    other = index_batch(params, torch.arange(k, n_images * k + k, k))
    rel_null, _ = orbit_distance(first, other)

    return {
        "dataset": dataset,
        "n_pairs": int(n_images),
        "K": k,
        "R_f": r_f.tolist(),
        "R_f_median": float(r_f.median()),
        "R_theta": rel.tolist(),
        "R_theta_median": float(rel.median()),
        "R_theta_min": float(rel.min()),
        "R_theta_null_median": float(rel_null.median()),
        "sweeps": diag.sweeps,
        "note": "same-image pairs are only approximately function-equal; see R_f",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["planted", "null", "teacher"])
    ap.add_argument("--widths", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    ap.add_argument("--n", type=int, default=16, help="networks per width")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grid-side", type=int, default=64)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--warm-eps", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--warm-steps", type=int, default=8000,
                    help="warm-start fit length; the pilot value that produced P-S4e-3/4")
    ap.add_argument("--warm-n", type=int, default=32)
    ap.add_argument("--n-by-width", nargs="*", default=[],
                    help="per-width student counts as W:N (e.g. 2:128 8:64 32:32); "
                         "cost is dominated by batch size, not width")
    ap.add_argument("--root", default="data/inrbench")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=str(OUT / "s4e.json"))
    args = ap.parse_args()
    n_by_width = {int(k): int(v) for k, v in (s.split(":") for s in args.n_by_width)}
    n_for = lambda w: n_by_width.get(w, args.n)  # noqa: E731

    report: dict = {
        "study": "S4e — deep identifiability falsification hunt (PO-2 conjecture 6.5)",
        "prereg": "docs/prereg/S4e.md",
        "config": vars(args),
        "arms": {},
    }

    if "planted" in args.arms:
        report["arms"]["planted"] = [
            arm_planted(w, n_for(w), args.device) for w in args.widths
        ]
        for r in report["arms"]["planted"]:
            print(f"planted  w={r['width']:3d}  R_theta max {r['R_theta_max']:.3e} "
                  f"({r['sweeps']} sweeps)", flush=True)

    if "warmstart" in args.arms:
        rows = [
            arm_warmstart(w, args.warm_n, args.warm_steps, args.lr, args.grid_side,
                          args.device, eps=e)
            for w in args.widths
            for e in args.warm_eps
        ]
        report["arms"]["warmstart"] = rows
        for r in rows:
            print(f"warmst.  w={r['width']:3d} eps={r['eps_start']:.0e}  "
                  f"R_f {r['R_f_start']:.1e}->{r['R_f_median']:.1e}  "
                  f"R_theta {r['R_theta_start']:.1e}->{r['R_theta_median']:.1e}  "
                  f"recovered {r['recovered_frac']:.0%}", flush=True)

    if "sensitivity" in args.arms:
        ladder = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
        rows = [arm_sensitivity(w, args.warm_n, args.grid_side, ladder) for w in args.widths]
        report["arms"]["sensitivity"] = rows
        for r in rows:
            k = [f"{x["kappa_median"]:.4f}" for x in r["ladder"]]
            print(f"sensit.  w={r['width']:3d}  kappa(median) over eps ladder: {k}", flush=True)

    if "null" in args.arms:
        report["arms"]["null"] = [arm_null(w, args.warm_n, args.device) for w in args.widths]
        for r in report["arms"]["null"]:
            print(f"null     w={r['width']:3d}  R_theta median {r['R_theta_median']:.3f} "
                  f"min {r['R_theta_min']:.3f}", flush=True)

    if "teacher" in args.arms:
        rows = []
        for w in args.widths:
            r = arm_teacher(w, n_for(w), args.steps, args.lr, args.grid_side, args.device)
            rows.append(r)
            print(f"teacher  w={w:3d}  best R_f {r['R_f_min']:.3e}  "
                  f"R_theta there {r['R_theta_at_best_R_f']:.4f}  "
                  f"median R_theta {r['R_theta_median']:.4f}  ({r['wallclock_s']:.0f}s)",
                  flush=True)
        report["arms"]["teacher"] = rows

    if "production" in args.arms:
        r = arm_production(args.dataset, args.n_images, args.device, Path(args.root))
        report["arms"]["production"] = r
        print(f"production {r['dataset']}: R_f median {r['R_f_median']:.4f}, "
              f"R_theta median {r['R_theta_median']:.4f} "
              f"(null {r['R_theta_null_median']:.4f})", flush=True)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

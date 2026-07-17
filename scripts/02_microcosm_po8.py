#!/usr/bin/env python3
"""PO-8 microcosm: exact profiled loss of the 1-neuron sine fit, numeric certification.

Deliverables (results/microcosm/):
  profiled_surface.png  — log10 L*(w,b) contour with the D_inf orbit marked (F4 left)
  basins.png            — init -> endpoint basin classification vs init range (F4 right)
  report.json           — zero-set check, spurious-minima census, basin counts,
                          Bessel-Vandermonde min |det| (PO-2-deep support)

Model u sin(wt+b)+c on [-1,1]; target A sin(omega t + phi) + c0. Linear params (u, c)
profiled out in closed form (ch2-fitmap.tex, Lemma 2.1); closed form cross-checked
against quadrature to 1e-9 before use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import jv

A, OMEGA, PHI, C0 = 1.0, 7.0, 0.8, 0.3
OUT = Path(__file__).resolve().parent.parent / "results" / "microcosm"


def sinc(x: np.ndarray) -> np.ndarray:
    return np.sinc(np.asarray(x) / np.pi)  # np.sinc is sin(pi x)/(pi x)


DET_EPS = 1e-9  # Gram near-singularity threshold: s(t) ~ constant (the w~0 ridge)


def profiled_loss(w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closed-form L*(w,b) = 0.5*(<y,y> - r^T G^+ r), vectorized.

    Where the 2x2 Gram G of {s, 1} is (near-)singular — s(t) ~ constant, i.e. the
    degenerate ridge w ~ 0 (and exact b = ±pi/2 lines at sin 2w = 2w... only w=0
    in practice) — the correct profiled loss is the best *constant* fit
    (pseudo-inverse semantics), not the blown-up G^{-1} formula.
    """
    w = np.asarray(w, dtype=float)
    b = np.asarray(b, dtype=float)
    ss = 1.0 - np.cos(2 * b) * sinc(2 * w)
    s1 = 2.0 * np.sin(b) * sinc(w)
    sy = A * (np.cos(b - PHI) * sinc(w - OMEGA) - np.cos(b + PHI) * sinc(w + OMEGA)) + C0 * s1
    oy = 2.0 * A * np.sin(PHI) * sinc(OMEGA) + 2.0 * C0
    yy = A * A * (1.0 - np.cos(2 * PHI) * sinc(2 * OMEGA)) + 4.0 * A * C0 * np.sin(PHI) * sinc(OMEGA) + 2.0 * C0 * C0
    det = ss * 2.0 - s1 * s1
    safe_det = np.where(det > DET_EPS, det, 1.0)
    quad_full = (2.0 * sy * sy - 2.0 * s1 * sy * oy + ss * oy * oy) / safe_det
    quad_const = oy * oy / 2.0  # projection onto span{1} only
    quad_form = np.where(det > DET_EPS, quad_full, quad_const)
    return 0.5 * (yy - quad_form)


def on_degenerate_ridge(w: float) -> bool:
    return abs(w) < 0.05


def quadrature_loss(w: float, b: float) -> float:
    def s(t):
        return np.sin(w * t + b)

    def y(t):
        return A * np.sin(OMEGA * t + PHI) + C0

    ss = quad(lambda t: s(t) ** 2, -1, 1)[0]
    s1 = quad(s, -1, 1)[0]
    sy = quad(lambda t: s(t) * y(t), -1, 1)[0]
    oy = quad(y, -1, 1)[0]
    yy = quad(lambda t: y(t) ** 2, -1, 1)[0]
    g = np.array([[ss, s1], [s1, 2.0]])
    r = np.array([sy, oy])
    return 0.5 * (yy - r @ np.linalg.solve(g, r))


def to_fundamental(w: float, b: float) -> tuple[float, float]:
    """Reduce (w,b) by sigma (w>0) and tau/rho (b in [-pi/2, pi/2))."""
    if w < 0:
        w, b = -w, -b
    b = b - np.floor(b / np.pi + 0.5) * np.pi
    return w, b


def orbit_distance(w: float, b: float) -> float:
    """Distance of (w,b) to the global-minimum orbit {(±omega, ±phi mod pi)}."""
    wf, bf = to_fundamental(w, b)
    _, phif = to_fundamental(OMEGA, PHI)
    db = abs(bf - phif)
    return float(np.hypot(wf - OMEGA, min(db, np.pi - db)))


def find_local_minima(grid_w: np.ndarray, grid_b: np.ndarray, vals: np.ndarray) -> list[dict]:
    """Grid-local minima refined by Nelder-Mead; deduped in the fundamental domain."""
    minima: list[dict] = []
    interior = np.ones_like(vals, dtype=bool)
    for dw in (-1, 0, 1):
        for db in (-1, 0, 1):
            if dw == db == 0:
                continue
            interior[1:-1, 1:-1] &= vals[1:-1, 1:-1] <= np.roll(np.roll(vals, dw, 0), db, 1)[1:-1, 1:-1]
    cand = np.argwhere(interior)
    seen: list[tuple[float, float]] = []
    for i, j in cand:
        res = minimize(
            lambda p: float(profiled_loss(p[0], p[1])),
            x0=[grid_w[i], grid_b[j]],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 2000},
        )
        wf, bf = to_fundamental(res.x[0], res.x[1])
        if wf < 1e-3:  # w ~ 0 ridge (constant neuron) — not an isolated minimum
            continue
        if any(np.hypot(wf - a, bf - c) < 1e-3 for a, c in seen):
            continue
        seen.append((wf, bf))
        minima.append(
            {
                "w": round(wf, 6),
                "b": round(bf, 6),
                "loss": float(res.fun),
                "type": "global(orbit)" if res.fun < 1e-10 else "spurious",
                "orbit_dist": round(orbit_distance(wf, bf), 6),
            }
        )
    return sorted(minima, key=lambda m: m["loss"])


def basin_census(init_ranges: list[float], n_inits: int = 900, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for w0 in init_ranges:
        endpoints = []
        for _ in range(n_inits):
            x0 = [rng.uniform(-w0, w0), rng.uniform(-np.pi, np.pi)]
            res = minimize(
                lambda p: float(profiled_loss(p[0], p[1])),
                x0=x0,
                method="Nelder-Mead",
                options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 3000},
            )
            wf, bf = to_fundamental(res.x[0], res.x[1])
            endpoints.append((wf, bf, float(res.fun), x0))
        clusters: list[tuple[float, float, float]] = []
        labels = []
        for wf, bf, fv, _ in endpoints:
            for ci, (cw, cb, _) in enumerate(clusters):
                if np.hypot(wf - cw, bf - cb) < 5e-2:
                    labels.append(ci)
                    break
            else:
                clusters.append((wf, bf, fv))
                labels.append(len(clusters) - 1)
        def classify(wf: float, bf: float, fv: float) -> str:
            if on_degenerate_ridge(wf):
                return "ridge"  # constant-fit plateau at w ~ 0 (Gram-degenerate)
            if fv < 1e-9 or orbit_distance(wf, bf) < 5e-2:
                return "global"
            return "spurious"

        cluster_class = [classify(cw, cb, fv) for (cw, cb, fv) in clusters]
        point_class = [classify(wf, bf, fv) for wf, bf, fv, _ in endpoints]
        out[str(w0)] = {
            "n_distinct_endpoints_fundamental": len(clusters),
            "n_global_reached": cluster_class.count("global"),
            "n_spurious_basins_hit": cluster_class.count("spurious"),
            "n_ridge_clusters": cluster_class.count("ridge"),
            "frac_inits_reaching_global": float(np.mean([c == "global" for c in point_class])),
            "frac_inits_on_ridge": float(np.mean([c == "ridge" for c in point_class])),
            "frac_inits_spurious": float(np.mean([c == "spurious" for c in point_class])),
            "_endpoints": endpoints,
            "_labels": labels,
            "_class": point_class,
        }
    return out


def bessel_vandermonde_check(trials: int = 200, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    worst = {}
    for n in (2, 3, 4, 5):
        dets = []
        for _ in range(trials):
            x = np.sort(rng.uniform(0.1, 3.0, size=n))
            if np.min(np.diff(x)) < 1e-3:
                continue
            m = np.array([[jv(k, xj) for xj in x] for k in range(n)])
            dets.append(abs(np.linalg.det(m)))
        worst[str(n)] = float(min(dets))
    return worst


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"params": {"A": A, "omega": OMEGA, "phi": PHI, "c0": C0}}

    # 0) closed form vs quadrature
    rng = np.random.default_rng(3)
    errs = []
    for _ in range(40):
        w, b = rng.uniform(0.5, 15), rng.uniform(-np.pi, np.pi)
        errs.append(abs(profiled_loss(w, b) - quadrature_loss(w, b)))
    report["closed_form_max_abs_err_vs_quadrature"] = float(max(errs))
    assert report["closed_form_max_abs_err_vs_quadrature"] < 1e-9

    # 1) zero set = orbit (Prop 2.2)
    orbit_pts = [(OMEGA, PHI), (OMEGA, PHI - np.pi), (-OMEGA, -PHI), (-OMEGA, -PHI + np.pi)]
    report["loss_at_orbit_points"] = [float(profiled_loss(w, b)) for w, b in orbit_pts]
    assert max(report["loss_at_orbit_points"]) < 1e-12

    # 2) surface + minima census
    gw = np.linspace(0.05, 20, 1600)
    gb = np.linspace(-np.pi / 2, np.pi / 2, 400, endpoint=False)
    vals = profiled_loss(gw[:, None], gb[None, :])
    minima = find_local_minima(gw, gb, vals)
    report["n_minima_fundamental_domain"] = len(minima)
    report["n_spurious"] = sum(1 for m in minima if m["type"] == "spurious")
    report["minima"] = minima[:40]

    # 3) basin census vs init range
    census = basin_census([2.0, 5.0, 10.0, 20.0])
    report["basin_census"] = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in census.items()
    }

    # 4) Bessel-Vandermonde support for PO-2-deep
    report["bessel_vandermonde_min_abs_det"] = bessel_vandermonde_check()

    # figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.pcolormesh(gw, gb, np.log10(np.maximum(vals.T, 1e-16)), shading="auto", cmap="viridis")
    for w, b in orbit_pts:
        wf, bf = to_fundamental(w, b)
        ax.plot(wf, bf, "r*", ms=14, mec="white")
    for m in minima:
        if m["type"] == "spurious":
            ax.plot(m["w"], m["b"], "wx", ms=6)
    ax.set(xlabel="w", ylabel="b (fundamental domain)", title=f"log10 L*(w,b) — target ω={OMEGA}, φ={PHI}; ★ D∞ orbit, × spurious minima")
    fig.colorbar(im, ax=ax, label="log10 L*")
    fig.tight_layout()
    fig.savefig(OUT / "profiled_surface.png", dpi=160)

    fig2, axes = plt.subplots(1, len(census), figsize=(4 * len(census), 3.6), sharey=True)
    colors = {"global": "#2a9d8f", "spurious": "#e76f51", "ridge": "#8d99ae"}
    for ax2, (w0, data) in zip(axes, census.items()):
        pts = data["_endpoints"]
        cls = data["_class"]
        for name, col in colors.items():
            sel = [p for p, c in zip(pts, cls) if c == name]
            if sel:
                ax2.scatter([p[3][0] for p in sel], [p[3][1] for p in sel], s=4, c=col, label=f"→ {name}")
        ax2.set(
            title=f"±{w0}: {data['n_spurious_basins_hit']} spurious, ridge {data['frac_inits_on_ridge']:.0%}",
            xlabel="init w",
        )
    axes[0].set_ylabel("init b")
    axes[0].legend(loc="upper left", fontsize=7)
    fig2.suptitle("Gradient-descent basin census (endpoint class by init)")
    fig2.tight_layout()
    fig2.savefig(OUT / "basins.png", dpi=160)

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("minima",)}, indent=2)[:2500])
    print(f"\nminima found: {len(minima)} (spurious: {report['n_spurious']}); artifacts in {OUT}")


if __name__ == "__main__":
    sys.exit(main())

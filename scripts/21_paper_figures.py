#!/usr/bin/env python3
"""Publication figures for the paper and the README.

Reads only committed result artifacts (results/ladder/*, results/microcosm/*,
docs/PREDICTION_OUTCOMES.csv) and writes vector PDF (for LaTeX) plus PNG (for the
README) into paper/figures/. Datasets that have not been run yet are skipped, so the
script is safe to run while a ladder is still in flight and again after it lands.

Usage:
  .venv/bin/python scripts/21_paper_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / "results" / "ladder"
MICRO = ROOT / "results" / "microcosm"
OUT = ROOT / "paper" / "figures"

DATASETS = [
    ("mnist", "MNIST", "#3B6EA5"),
    ("fashionmnist", "FashionMNIST", "#C4622D"),
    ("cifar10", "CIFAR-10", "#4E9A6A"),
]

# display order: ceiling controls, then the ladder sorted by what it recovers
ACC_ORDER = ["P0", "P1", "W1", "W2", "W5", "W10", "W4", "W6", "W7", "W9", "W3", "W8"]
F_ORDER = ["W5", "W10", "W4", "W6", "W7", "W9"]

RUNG_LABEL = {
    "P0": "P0  real pixels",
    "P1": "P1  oracle render",
    "W1": "W1  weights, shared init",
    "W2": "W2  weights, shared init + SGD noise",
    "W3": "W3  weights, random init",
    "W4": r"W4  $c_{\mathrm{sort}}$ (template-free)",
    "W5": r"W5  $c_{\mathrm{align}}$ (to $\theta_0$)",
    "W6": "W6  group augmentation",
    "W7": "W7  $K$-marginalization",
    "W8": "W8  canonicalize + augment",
    "W9": "W9  frame averaging",
    "W10": "W10  exact $L{=}2$ invariants",
}
F_LABEL = {
    "W4": "W4\nsort",
    "W5": "W5\nalign",
    "W6": "W6\naugment",
    "W7": "W7\nmarg.",
    "W9": "W9\nframe",
    "W10": "W10\ninvar.",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png")


def load_means(dataset: str) -> dict[str, float] | None:
    """Prefer the analysis file; fall back to per-rung cells for an in-flight ladder."""
    ana = LADDER / dataset / "S1_analysis.json"
    if ana.exists():
        return json.loads(ana.read_text())["means"]
    cells = sorted((LADDER / dataset).glob("*.json")) if (LADDER / dataset).exists() else []
    means = {}
    for c in cells:
        if c.name.startswith(("S1_", "EXPLORATORY")):
            continue
        d = json.loads(c.read_text())
        if "mean" in d and "rung" in d:
            means[d["rung"]] = d["mean"]
    return means or None


def recovery(means: dict[str, float], rung: str) -> float | None:
    if not {"W1", "W3", rung} <= means.keys():
        return None
    denom = means["W1"] - means["W3"]
    return (means[rung] - means["W3"]) / denom if abs(denom) > 1e-9 else None


# --------------------------------------------------------------------------- F1


def fig_ladder(available: list[tuple[str, str, str, dict]]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.9, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) absolute accuracy per rung
    rows = [r for r in ACC_ORDER if any(r in m for *_, m in available)]
    y = np.arange(len(rows))[::-1]
    h = 0.8 / len(available)
    for k, (_, label, color, means) in enumerate(available):
        vals = [means.get(r, np.nan) for r in rows]
        ax0.barh(y + (k - (len(available) - 1) / 2) * h, vals, height=h * 0.92,
                 color=color, label=label, linewidth=0)
    ax0.axvline(10, color="0.35", lw=0.7, ls=(0, (3, 2)))
    ax0.text(10.8, y[-1] - 0.05, "chance", fontsize=6.2, color="0.35", va="center")
    ax0.set_yticks(y)
    ax0.set_yticklabels([RUNG_LABEL[r] for r in rows])
    ax0.set_xlabel("test accuracy (\\%)" if matplotlib.rcParams["text.usetex"] else "test accuracy (%)")
    ax0.set_xlim(0, 100)
    ax0.set_title("(a)  the ladder: what a matched reader recovers", loc="left")
    ax0.legend(frameon=False, loc="lower right", handlelength=1.1)
    ax0.tick_params(axis="y", length=0)

    # (b) recovery fraction against signal complexity — the crossover
    methods = [
        ("W5", r"W5  $\calign$ (alignment)", "#3B6EA5", "o", "-"),
        ("W10", r"W10  exact $L{=}2$ invariants", "#4E9A6A", "D", "-"),
        ("W4", r"W4  $\csort$ (template-free)", "#C4622D", "s", "-"),
        ("W6", "W6  augmentation", "#9A9A9A", "^", (0, (3, 2))),
        ("W7", "W7  $K$-marginalization", "#B8B8B8", "v", (0, (3, 2))),
        ("W9", "W9  frame averaging", "#CFCFCF", "x", (0, (1, 2))),
    ]
    x = np.arange(len(available))
    for rung, label, color, marker, ls in methods:
        vals = [recovery(m, rung) for *_, m in available]
        if all(v is None for v in vals):
            continue
        ax1.plot(x, [np.nan if v is None else v for v in vals], marker=marker, ms=4.0,
                 lw=1.2, ls=ls, color=color, label=label.replace(r"$\calign$", r"$c_{\mathrm{align}}$")
                 .replace(r"$\csort$", r"$c_{\mathrm{sort}}$"))
    ax1.axhline(0, color="0.2", lw=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl, _, _ in available], fontsize=7)
    ax1.set_xlim(-0.25, len(available) - 0.75)
    ax1.set_ylabel(r"recovery fraction $f$")
    ax1.set_ylim(-0.06, 0.88)
    ax1.set_title("(b)  what survives the change of signal", loc="left")
    ax1.legend(frameon=False, loc="upper center", handlelength=1.8, ncol=1,
               labelspacing=0.25, borderpad=0.1)

    fig.tight_layout(w_pad=1.6)
    save(fig, "fig1_ladder")


# --------------------------------------------------------------------------- F2


def fig_mechanism() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    micro = __import__("02_microcosm_po8")

    census = json.loads((MICRO / "optimizer_census.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.15))

    # (a) profiled loss surface with the D_infty orbit of global minima
    ax = axes[0]
    ws = np.linspace(-14, 14, 420)
    bs = np.linspace(-np.pi, np.pi, 380)
    W, B = np.meshgrid(ws, bs, indexing="ij")
    Z = micro.profiled_loss(W, B)
    im = ax.pcolormesh(W, B, np.log10(np.maximum(Z, 1e-12)), cmap="magma", shading="auto",
                       rasterized=True)
    omega, phi = 7.0, 0.8
    orb_w, orb_b = [], []
    for eps in (1, -1):
        for k in range(-2, 3):
            b = eps * phi + np.pi * k
            if -np.pi <= b <= np.pi:
                orb_w.append(eps * omega)
                orb_b.append(b)
    ax.scatter(orb_w, orb_b, s=13, facecolor="none", edgecolor="#5ee0d0", linewidth=0.9,
               label=r"$D_\infty$ orbit")
    ax.set_xlabel(r"frequency $w$")
    ax.set_ylabel(r"phase $b$")
    ax.set_title(r"(a)  profiled loss $\mathcal{L}^*(w,b)$", loc="left")
    ax.legend(frameon=True, framealpha=0.85, edgecolor="none", loc="upper right",
              handletextpad=0.3, borderpad=0.25)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"$\log_{10}\mathcal{L}^*$", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    # (b) global-capture fraction vs init range, three optimizers + the Nelder-Mead reference
    ax = axes[1]
    series = [
        ("nelder_mead_reference", "Nelder–Mead (profiled)", "#8C8C8C", "o", (0, (2, 1.5))),
        ("adam_converged", "Adam, converged", "#3B6EA5", "s", "-"),
        ("gd_converged", "GD, converged", "#C4622D", "^", "-"),
        ("adam_production", "Adam, corpus setting", "#4E9A6A", "D", "-"),
    ]
    for key, label, color, marker, ls in series:
        src = census[key] if key == "nelder_mead_reference" else census["census"][key]["by_init_range"]
        ranges = sorted(float(r) for r in src)
        vals = [src[f"{r}"]["frac_inits_reaching_global"] for r in ranges]
        ax.plot(ranges, vals, marker=marker, ms=3.2, lw=1.0, ls=ls, color=color, label=label)
    ax.axvline(7.0, color="0.35", lw=0.7, ls=(0, (3, 2)))
    ax.text(7.3, 0.63, r"$\omega=7$", fontsize=6.4, color="0.35")
    ax.set_xlabel("initialization range")
    ax.set_ylabel("fraction reaching the global orbit")
    ax.set_title("(b)  basin capture is non-monotone", loc="left")
    ax.set_ylim(-0.03, 0.72)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5)

    # (c) laziness at the corpus setting
    ax = axes[2]
    prod = census["census"]["adam_production"]["by_init_range"]
    conv = census["census"]["adam_converged"]["by_init_range"]
    ranges = sorted(float(r) for r in prod)
    ax.plot(ranges, [prod[f"{r}"]["median_abs_w_travel"] for r in ranges],
            marker="D", ms=3.2, lw=1.0, color="#4E9A6A", label="corpus setting (300 steps)")
    ax.plot(ranges, [conv[f"{r}"]["median_abs_w_travel"] for r in ranges],
            marker="s", ms=3.2, lw=1.0, color="#3B6EA5", label="converged (5000 steps)")
    ax.set_xlabel("initialization range")
    ax.set_ylabel(r"median $|\Delta w|$ from init")
    ax.set_title("(c)  the fit does not leave its init", loc="left")
    ax.set_ylim(0, 1.85)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig2_mechanism")


# --------------------------------------------------------------------------- F3


def fig_calibration() -> None:
    rows = list(csv.DictReader((ROOT / "docs" / "PREDICTION_OUTCOMES.csv").open()))
    iv = [r for r in rows if r["kind"] == "interval"]
    # normalize each prediction to its own registered interval: 0 = lo80, 1 = hi80
    names, z_obs, hits = [], [], []
    for r in iv:
        lo, hi, obs = float(r["lo80"]), float(r["hi80"]), float(r["observed"])
        span = hi - lo
        names.append(r["prediction"].split(" ")[0])
        z_obs.append((obs - lo) / span if span else 0.0)
        hits.append(r["verdict"] == "HIT")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.9, 2.6), gridspec_kw={"width_ratios": [1.7, 1]})

    y = np.arange(len(names))[::-1]
    ax0.axvspan(0, 1, color="#3B6EA5", alpha=0.12, lw=0)
    ax0.axvline(0, color="0.5", lw=0.6)
    ax0.axvline(1, color="0.5", lw=0.6)
    for yi, z, hit in zip(y, z_obs, hits):
        zc = np.clip(z, -0.65, 1.85)
        ax0.plot([0.5, zc], [yi, yi], color="0.75", lw=0.7, zorder=1)
        ax0.scatter([zc], [yi], s=17, zorder=2,
                    color="#3B6EA5" if hit else "#C0392B",
                    marker="o" if abs(z - zc) < 1e-9 else ">")
    ax0.set_yticks(y)
    ax0.set_yticklabels(names)
    ax0.set_xlim(-0.75, 1.95)
    ax0.set_xticks([0, 0.5, 1])
    ax0.set_xticklabels(["lo80", "point", "hi80"])
    ax0.set_xlabel("observation, in units of its own registered 80\\% interval"
                   if matplotlib.rcParams["text.usetex"]
                   else "observation, in units of its own registered 80% interval")
    ax0.set_title("(a)  every registered interval, scored", loc="left")
    ax0.tick_params(axis="y", length=0)
    ax0.text(1.9, y[-1] - 0.1, "arrow = off scale", fontsize=6, ha="right", color="0.45")

    cov = float(np.mean(hits))
    ax1.bar([0, 1], [0.80, cov], width=0.55, color=["#B8C6D9", "#3B6EA5"], linewidth=0)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["nominal", f"observed\n({sum(hits)}/{len(hits)})"])
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("80\\% interval coverage" if matplotlib.rcParams["text.usetex"]
                   else "80% interval coverage")
    ax1.axhline(0.80, color="0.4", lw=0.7, ls=(0, (3, 2)))
    ax1.set_title("(b)  the intervals are too narrow", loc="left")
    for xi, v in zip([0, 1], [0.80, cov]):
        ax1.text(xi, v + 0.025, f"{v:.0%}", ha="center", fontsize=7)

    fig.tight_layout(w_pad=1.6)
    save(fig, "fig3_calibration")


# --------------------------------------------------------------------------- F4


def fig_template() -> None:
    path = LADDER / "mnist" / "EXPLORATORY_w5_template_sensitivity.json"
    if not path.exists():
        print("skip fig4: no template sensitivity file")
        return
    d = json.loads(path.read_text())
    arms = d.get("templates")
    if not arms:
        print(f"skip fig4: unexpected schema {list(d)}")
        return
    pretty = {
        "theta0_shared_init": r"$\theta_0$: the corpus's shared init",
        "unrelated_init_12345": "unrelated random init (seed 12345)",
        "unrelated_init_777": "unrelated random init (seed 777)",
        "a_fitted_shared_det_inr": "a fitted INR (shared-init corpus)",
        "a_fitted_random_inr": "a fitted INR (random-init corpus)",
    }
    items = sorted(arms.items(), key=lambda kv: -kv[1]["recovery_fraction"])
    labels = [pretty.get(k, k) for k, _ in items]
    fvals = [v["recovery_fraction"] for _, v in items]

    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    colors = ["#3B6EA5" if r"\theta_0" in lbl else "#9BB2CB" for lbl in labels]
    ax.barh(np.arange(len(fvals))[::-1], fvals, height=0.66, color=colors, linewidth=0)
    ax.axvline(0.5, color="#C0392B", lw=0.8, ls=(0, (3, 2)))
    ax.text(0.51, -0.15, "prereg falsification line", fontsize=6, color="#C0392B")
    ax.set_yticks(np.arange(len(fvals))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"recovery fraction $f(\mathrm{W5})$")
    ax.set_xlim(0, 0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_title("alignment template does not matter", loc="left")
    fig.tight_layout()
    save(fig, "fig4_template")


def main() -> None:
    style()
    available = []
    for key, label, color in DATASETS:
        means = load_means(key)
        if means and {"W1", "W3"} <= means.keys():
            available.append((key, label, color, means))
            print(f"{key}: {len(means)} rungs")
        else:
            print(f"{key}: skipped (ladder not complete)")
    if available:
        fig_ladder(available)
    fig_mechanism()
    fig_calibration()
    fig_template()


if __name__ == "__main__":
    main()

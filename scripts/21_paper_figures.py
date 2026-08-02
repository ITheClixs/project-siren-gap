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
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.5, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) absolute accuracy per rung
    rows = [r for r in ACC_ORDER if any(r in m for *_, m in available)]
    y = np.arange(len(rows))[::-1]
    h = 0.8 / len(available)
    for k, (_, label, color, means) in enumerate(available):
        vals = [means.get(r, np.nan) for r in rows]
        ax0.barh(y + (k - (len(available) - 1) / 2) * h, vals, height=h * 0.92,
                 color=color, label=label, linewidth=0)
    ax0.axvline(10, color="0.35", lw=0.7, ls=(0, (3, 2)))
    ax0.text(10.8, y[0] + 0.62, "chance", fontsize=6.2, color="0.35", va="center")
    ax0.set_yticks(y)
    ax0.set_yticklabels([RUNG_LABEL[r] for r in rows])
    ax0.set_xlabel("test accuracy (\\%)" if matplotlib.rcParams["text.usetex"] else "test accuracy (%)")
    ax0.set_xlim(0, 100)
    ax0.set_title("(a)  what a matched reader recovers", loc="left")
    ax0.legend(frameon=False, loc="lower right", handlelength=1.1, fontsize=6.6)
    ax0.tick_params(axis="y", length=0)

    # (b) recovery fraction against signal complexity — the crossover
    exact = [
        ("W5", r"W5  $c_{\mathrm{align}}$", "#3B6EA5", "o"),
        ("W10", r"W10  invariants", "#4E9A6A", "D"),
        ("W4", r"W4  $c_{\mathrm{sort}}$", "#C4622D", "s"),
    ]
    x = np.arange(len(available))
    for rung, _, color, marker in [("W6", "", "#A8A8A8", "^"), ("W7", "", "#BEBEBE", "v"),
                                   ("W9", "", "#D4D4D4", "x")]:
        vals = [recovery(m, rung) for *_, m in available]
        if all(v is None for v in vals):
            continue
        ax1.plot(x, [np.nan if v is None else v for v in vals], marker=marker, ms=3.2,
                 lw=1.0, ls=(0, (3, 2)), color=color, zorder=1)
    for rung, label, color, marker in exact:
        vals = [recovery(m, rung) for *_, m in available]
        if all(v is None for v in vals):
            continue
        ax1.plot(x, [np.nan if v is None else v for v in vals], marker=marker, ms=4.2,
                 lw=1.4, color=color, label=label, zorder=3)
    ax1.axhline(0, color="0.2", lw=0.6)
    ax1.annotate("W6, W7, W9\n(inexact treatments)", xy=(x[-1] - 0.05, 0.09),
                 xytext=(x[-1] - 0.75, 0.235), fontsize=6.0, color="0.45", ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="0.6"))
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl.replace("FashionMNIST", "Fashion-\nMNIST") for _, lbl, _, _ in available],
                        fontsize=6.6)
    ax1.set_xlim(-0.25, len(available) - 0.75)
    ax1.set_ylabel(r"recovery fraction $f$")
    ax1.set_ylim(-0.06, 0.95)
    ax1.set_title("(b)  the crossover", loc="left")
    ax1.legend(frameon=False, loc="upper left", handlelength=1.4, labelspacing=0.2,
               borderpad=0.0, fontsize=6.6)

    fig.tight_layout(w_pad=1.6)
    save(fig, "fig1_ladder")


# --------------------------------------------------------------------------- F2


def fig_mechanism() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    micro = __import__("02_microcosm_po8")

    census = json.loads((MICRO / "optimizer_census.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))

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
    ax.set_title(r"(a)  profiled loss", loc="left")
    ax.legend(frameon=True, framealpha=0.85, edgecolor="none", loc="upper right",
              handletextpad=0.3, borderpad=0.25)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"$\log_{10}\mathcal{L}^*$", fontsize=6)
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
    ax.text(7.3, 0.03, r"$\omega=7$", fontsize=6.4, color="0.35")
    ax.set_xlabel("initialization range")
    ax.set_ylabel("fraction reaching global orbit", fontsize=7)
    ax.set_title("(b)  basin capture", loc="left")
    ax.set_ylim(-0.03, 0.92)
    ax.legend(frameon=False, loc="upper left", handlelength=1.3, fontsize=5.4,
              labelspacing=0.18, borderpad=0.0)

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
    ax.set_title("(c)  travel from init", loc="left")
    ax.set_ylim(0, 2.05)
    ax.legend(frameon=False, loc="upper left", handlelength=1.3, fontsize=5.4,
              labelspacing=0.18, borderpad=0.0)

    fig.tight_layout(w_pad=1.9)
    save(fig, "fig2_mechanism")


# --------------------------------------------------------------------------- F3


def fig_calibration() -> None:
    rows = list(csv.DictReader((ROOT / "docs" / "PREDICTION_OUTCOMES.csv").open()))
    iv = [r for r in rows if r["kind"] == "interval"]
    # normalize each prediction to its own registered interval: 0 = lo80, 1 = hi80
    names, z_obs, hits, arm = [], [], [], []
    for r in iv:
        lo, hi, obs = float(r["lo80"]), float(r["hi80"]), float(r["observed"])
        span = hi - lo
        pid = r["prediction"].split(" ")[0]
        names.append(pid)
        z_obs.append((obs - lo) / span if span else 0.0)
        hits.append(r["verdict"] == "HIT")
        arm.append("cifar" if pid.startswith("H-C1") else "grayscale")

    n_c = sum(a == "cifar" for a in arm)
    n_g = len(arm) - n_c
    hits_g = [h for h, a in zip(hits, arm) if a == "grayscale"]
    hits_c = [h for h, a in zip(hits, arm) if a == "cifar"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.85, 1]})

    y = np.arange(len(names))[::-1]
    ax0.axvspan(0, 1, color="#3B6EA5", alpha=0.12, lw=0)
    ax0.axvline(0, color="0.5", lw=0.6)
    ax0.axvline(1, color="0.5", lw=0.6)
    for yi, z, hit in zip(y, z_obs, hits):
        zc = float(np.clip(z, -0.65, 1.85))
        ax0.plot([0.5, zc], [yi, yi], color="0.78", lw=0.6, zorder=1)
        ax0.scatter([zc], [yi], s=14, zorder=2,
                    color="#3B6EA5" if hit else "#C0392B",
                    marker="o" if abs(z - zc) < 1e-9 else ">")
    if n_c:
        div = y[n_g] + 0.5
        ax0.axhline(div, color="0.35", lw=0.6, ls=(0, (2, 2)))
        ax0.text(-0.72, div + 0.55, f"grayscale arms — {sum(hits_g)}/{n_g}",
                 fontsize=6, color="0.3", va="bottom")
        ax0.text(-0.72, div - 0.55, f"CIFAR-10 arm — {sum(hits_c)}/{n_c}",
                 fontsize=6, color="0.3", va="top")
    ax0.set_yticks(y)
    ax0.set_yticklabels(names, fontsize=5.4)
    ax0.set_ylim(y[-1] - 0.8, y[0] + 1.8)
    ax0.set_xlim(-0.75, 1.95)
    ax0.set_xticks([0, 0.5, 1])
    ax0.set_xticklabels(["lo80", "point", "hi80"])
    ax0.set_xlabel("observation, in units of its own registered 80% interval")
    ax0.set_title("(a)  every registered interval, scored", loc="left")
    ax0.tick_params(axis="y", length=0)
    ax0.text(1.92, y[0] + 1.15, "arrow = off scale", fontsize=5.6, ha="right", color="0.45")

    bars = [("nominal", 0.80, "#B8C6D9"),
            (f"grayscale\n({sum(hits_g)}/{n_g})", float(np.mean(hits_g)), "#9BB2CB")]
    if n_c:
        bars.append((f"CIFAR-10\n({sum(hits_c)}/{n_c})", float(np.mean(hits_c)), "#3B6EA5"))
    ax1.bar(range(len(bars)), [b[1] for b in bars], width=0.6,
            color=[b[2] for b in bars], linewidth=0)
    ax1.set_xticks(range(len(bars)))
    ax1.set_xticklabels([b[0] for b in bars], fontsize=6.4)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("80% interval coverage")
    ax1.axhline(0.80, color="0.4", lw=0.7, ls=(0, (3, 2)))
    ax1.set_title("(b)  calibration, before and after", loc="left")
    for xi, b in enumerate(bars):
        ax1.text(xi, b[1] + 0.02, f"{b[1]:.0%}", ha="center", fontsize=7)

    fig.tight_layout(w_pad=1.4)
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

    fig, ax = plt.subplots(figsize=(3.5, 2.1))
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


def fig_s4e() -> None:
    """S4e: does functional near-equality imply parameter proximity modulo the group?"""
    path = ROOT / "results" / "s4e" / "s4e.json"
    if not path.exists():
        print("skip fig5: no S4e results")
        return
    arms = json.loads(path.read_text())["arms"]
    if not {"teacher", "warmstart", "sensitivity", "null"} <= arms.keys():
        print("skip fig5: S4e run incomplete")
        return

    widths = sorted(r["width"] for r in arms["teacher"])
    cmap = plt.get_cmap("viridis")
    colour = {w: cmap(i / max(len(widths) - 1, 1) * 0.85) for i, w in enumerate(widths)}

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(5.5, 2.0),
                                        gridspec_kw={"width_ratios": [1.35, 1, 0.85]})

    # (a) the scatter: independent students, R_theta against R_f
    null_med = float(np.median([r["R_theta_median"] for r in arms["null"]]))
    kap = {r["width"]: r["ladder"][0]["kappa_median"] for r in arms["sensitivity"]}
    ax0.axhspan(null_med * 0.8, null_med * 1.25, color="#C0392B", alpha=0.10, lw=0)
    ax0.text(3e-8, null_med * 1.35, "unrelated networks", fontsize=5.6, color="#8E2B20")
    xs = np.logspace(-8, 0.5, 60)
    k_ref = float(np.median(list(kap.values())))
    ax0.plot(xs, k_ref * xs, color="0.35", lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax0.text(0.11, 0.9e-3, r"$R_\theta=\kappa R_f$" "\n(local conditioning)",
             fontsize=5.6, color="0.3", ha="left")
    ax0.annotate("recovered exactly", xy=(7e-8, 1.6e-7), xytext=(2.5e-7, 4e-5),
                 fontsize=5.8, color="#1B5E20", ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.6, color="#2E7D32"))
    for row in arms["teacher"]:
        w = row["width"]
        ax0.scatter(row["R_f"], row["R_theta"], s=5, alpha=0.55, linewidth=0,
                    color=colour[w], label=f"$n{{=}}{w}$")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(2e-8, 6.0)
    ax0.set_ylim(2e-8, 6.0)
    ax0.set_xlabel(r"functional residual $R_f$")
    ax0.set_ylabel(r"orbit residual $R_\theta$")
    ax0.set_title("(a)  independent students", loc="left")
    ax0.legend(frameon=False, fontsize=5.4, loc="upper left", handletextpad=0.1,
               labelspacing=0.15, borderpad=0.0, ncol=2, columnspacing=0.6,
               bbox_to_anchor=(0.0, 0.86))

    # (b) the basin radius: warm-start recovery against the starting perturbation
    eps_vals = sorted({r["eps_start"] for r in arms["warmstart"]})
    for w in widths:
        ys = [
            next((r["recovered_frac"] for r in arms["warmstart"]
                  if r["width"] == w and r["eps_start"] == e), np.nan)
            for e in eps_vals
        ]
        ax1.plot(eps_vals, ys, marker="o", ms=3.0, lw=1.1, color=colour[w], label=f"$n{{=}}{w}$")
    bpath = ROOT / "results" / "s4e" / "budget_control.json"
    if bpath.exists():
        bc = json.loads(bpath.read_text())["arms"]["warmstart"]
        for w in sorted({r["width"] for r in bc}):
            pts = sorted((r["eps_start"], r["recovered_frac"]) for r in bc if r["width"] == w)
            ax1.plot([e for e, _ in pts], [v for _, v in pts], marker="s", ms=3.4, lw=0,
                     mfc="none", mec=colour[w], mew=0.9, zorder=4)
        ax1.plot([], [], marker="s", ms=3.4, lw=0, mfc="none", mec="0.35", mew=0.9,
                 label=r"$5\times$ steps")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"start distance $\varepsilon$ from the orbit")
    ax1.set_ylabel("fraction returning")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("(b)  basin of the true orbit", loc="left")
    ax1.legend(frameon=False, fontsize=5.2, loc="lower left", labelspacing=0.12,
               borderpad=0.0, handletextpad=0.3, ncol=2, columnspacing=0.7)

    # (c) local conditioning against width
    ws = sorted(kap)
    ax2.plot(ws, [kap[w] for w in ws], marker="s", ms=3.2, lw=1.1, color="#3B6EA5")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(ws)
    ax2.set_xticklabels([str(w) for w in ws], fontsize=6)
    ax2.set_xlabel("width $n$")
    ax2.set_ylabel(r"$\kappa = R_\theta / R_f$")
    ax2.set_title("(c)  local conditioning", loc="left")

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig5_s4e")


def fig_channels() -> None:
    """Images or channels? Luminance CIFAR against the three primary corpora.

    Ordered by image complexity, with output-channel count annotated. If the CIFAR behaviour of
    W5 and W10 is about c, the luminance arm sits with the grayscale corpora; if it is about
    image statistics, it sits with RGB CIFAR.
    """
    order = [
        ("mnist", "MNIST", 1),
        ("fashionmnist", "Fashion-\nMNIST", 1),
        ("cifar10gray", "CIFAR-10\nluminance", 1),
        ("cifar10", "CIFAR-10\nRGB", 3),
    ]
    means = {k: load_means(k) for k, _, _ in order}
    if means.get("cifar10gray") is None or not {"W1", "W3", "W5"} <= (means["cifar10gray"] or {}).keys():
        print("skip fig6: luminance-CIFAR ladder not complete")
        return

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    x = np.arange(len(order))
    for rung, label, colour, marker in [
        ("W5", r"W5  $c_{\mathrm{align}}$", "#3B6EA5", "o"),
        ("W10", r"W10  invariants", "#4E9A6A", "D"),
        ("W4", r"W4  $c_{\mathrm{sort}}$", "#C4622D", "s"),
    ]:
        vals = [recovery(means[k], rung) if means[k] else None for k, _, _ in order]
        ax.plot(x, [np.nan if v is None else v for v in vals], marker=marker, ms=4.2,
                lw=1.4, color=colour, label=label)
    ax.axvspan(2.5, 3.5, color="0.85", alpha=0.45, lw=0)
    ax.text(3.0, 0.86, "$c=3$", fontsize=6.2, ha="center", color="0.35")
    ax.text(1.0, 0.86, "$c=1$", fontsize=6.2, ha="center", color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in order], fontsize=6.4)
    ax.set_xlim(-0.3, len(order) - 0.7)
    ax.set_ylim(-0.05, 0.95)
    ax.set_ylabel(r"recovery fraction $f$")
    ax.set_title("images or channels?", loc="left")
    ax.legend(frameon=False, loc="center left", fontsize=6.4, labelspacing=0.2, borderpad=0.0)
    fig.tight_layout()
    save(fig, "fig6_channels")


def fig_pareto() -> None:
    """S5: the FLOPs-accuracy frontier that adjudicates weight against function access."""
    path = ROOT / "results" / "s5" / "pareto_mnist.json"
    if not path.exists():
        print("skip fig7: no S5 results")
        return
    d = json.loads(path.read_text())
    fq = sorted(d["function_query"], key=lambda r: r["flops"])
    wt = [r for r in d["weight_access"] if r["acc"] is not None]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.5, 2.5),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    fx = [r["flops"] / 1e6 for r in fq]
    fy = [r["acc"] for r in fq]
    ax0.plot(fx, fy, "-o", ms=4, lw=1.4, color="#3B6EA5", label="function-query (learned probes)")
    for r in fq:
        ax0.annotate(f"K={r['n_probes']}", (r["flops"] / 1e6, r["acc"]), fontsize=5.4,
                     textcoords="offset points", xytext=(3, -7), color="#2A4F77")

    marks = {"W3 raw weights": "s", "W4 c_sort": "^", "W5 c_align": "D",
             "W10 invariants": "v", "W11a equivariant (raw)": "P",
             "W11b equivariant (invariant)": "X", "W1 shared-init (ceiling)": "*"}
    for r in wt:
        ceiling = r["name"].startswith("W1 ")
        ax0.scatter(r["flops"] / 1e6, r["acc"], s=40 if ceiling else 26,
                    marker=marks.get(r["name"], "o"),
                    color="0.6" if ceiling else "#C4622D", zorder=3,
                    label="_", edgecolor="none")
        ax0.annotate(r["name"].split()[0], (r["flops"] / 1e6, r["acc"]), fontsize=5.4,
                     textcoords="offset points", xytext=(4, 3),
                     color="0.45" if ceiling else "#8E4520")
    ax0.scatter([], [], marker="D", s=26, color="#C4622D", label="weight access (rungs)")
    ax0.scatter([], [], marker="*", s=40, color="0.6", label="shared-init ceiling (not available)")
    ax0.set_xscale("log")
    ax0.set_xlabel("MFLOP per INR (analytic, inference)")
    ax0.set_ylabel("test accuracy (%)")
    ax0.set_ylim(5, 103)
    ax0.set_title("(a)  the frontier", loc="left")
    ax0.legend(frameon=False, fontsize=5.6, loc="lower right", labelspacing=0.2, borderpad=0.0)

    # (b) amortization: does canonicalise-once rescue weight access?
    import numpy as _np
    from math import isnan  # noqa: F401
    w5 = next(r for r in wt if r["name"] == "W5 c_align")
    kstar = next(r for r in fq if r["acc"] > 64.41)
    T = _np.arange(1, 201)
    wcost = (w5["preprocess"] + T * (w5["flops"] - w5["preprocess"])) / 1e6
    fcost = T * kstar["flops"] / 1e6
    ax1.plot(T, wcost, lw=1.4, color="#C4622D", label="weight: canonicalize once,\ndecode $T$ times")
    ax1.plot(T, fcost, lw=1.4, color="#3B6EA5",
             label=f"function-query $K{{=}}{kstar['n_probes']}$,\nprobes per task")
    ax1.set_xlabel("downstream tasks $T$")
    ax1.set_ylabel("total MFLOP per INR")
    ax1.set_title("(b)  amortization does not rescue it", loc="left")
    ax1.legend(frameon=False, fontsize=5.8, loc="upper left", labelspacing=0.4, borderpad=0.0)

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig7_pareto")


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
    fig_s4e()
    fig_channels()
    fig_pareto()


if __name__ == "__main__":
    main()

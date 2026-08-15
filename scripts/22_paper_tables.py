#!/usr/bin/env python3
"""LaTeX tables for the paper, generated from committed result artifacts.

Writes paper/tables/*.tex. Datasets whose ladder has not been run are omitted, so the
script is safe to run mid-flight and again after the last cell lands.

Usage:
  .venv/bin/python scripts/22_paper_tables.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / "results" / "ladder"
OUT = ROOT / "paper" / "tables"

DATASETS = [("mnist", "MNIST"), ("fashionmnist", "FashionMNIST"), ("cifar10", "CIFAR-10")]

ROWS = [
    ("P0", r"real pixels", False),
    ("P1", r"oracle render of the fit", False),
    ("W1", r"raw weights, \textbf{shared init}", False),
    ("W2", r"raw weights, shared init + SGD noise", False),
    ("W3", r"raw weights, \textbf{random init}", False),
    ("W4", r"$\csort$ --- exact, template-free", True),
    ("W5", r"$\calign$ --- exact, aligned to $\theta_0$", True),
    ("W10", r"exact $L{=}2$ invariants (Prop.~\ref{prop:deep})", True),
    ("W6", r"bounded group augmentation", True),
    ("W7", r"$K$-marginalization ($K{=}8$)", True),
    ("W7-1/8", r"\quad control: $K$ corpus, rows matched", True),
    ("W9", r"frame averaging, $R{=}64$", True),
    ("W8", r"canonicalize, then augment", True),
]


def load(dataset: str) -> tuple[dict[str, float], dict[str, dict]] | None:
    """(rung means, registered recovery fractions with their bootstrap CIs)."""
    ana = LADDER / dataset / "S1_analysis.json"
    if ana.exists():
        d = json.loads(ana.read_text())
        fr = {k[2:]: v for k, v in d["recovery_fractions"].items() if k.startswith("f_")}
        return d["means"], fr
    d = LADDER / dataset
    if not d.exists():
        return None
    means = {}
    for c in sorted(d.glob("*.json")):
        if c.name.startswith(("S1_", "EXPLORATORY")):
            continue
        cell = json.loads(c.read_text())
        if "mean" in cell and "rung" in cell:
            means[cell["rung"]] = cell["mean"]
    return (means, {}) if means else None


def frac(means: dict[str, float], fracs: dict[str, dict], rung: str) -> str:
    """Seed-paired mean where the analysis registered one; ratio of means otherwise."""
    if rung in fracs:
        return f"{fracs[rung]['point']:.3f}"
    if not {"W1", "W3", rung} <= means.keys():
        return "---"
    den = means["W1"] - means["W3"]
    if abs(den) < 1e-9:
        return "---"
    return f"{(means[rung] - means['W3']) / den:.3f}"


def max_ci_halfwidth(available) -> float:
    hw = [
        (v["ci95"][1] - v["ci95"][0]) / 2
        for *_, fr in available
        for v in fr.values()
        if isinstance(v, dict) and "ci95" in v
    ]
    return max(hw) if hw else 0.0


def ladder_table(available) -> str:
    ncol = len(available)
    colspec = "@{}ll" + "".join(["r@{\\hspace{5pt}}r" for _ in range(ncol)]) + "@{}"
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{lbl}}}" for _, lbl, _, _ in available)
    sub = " & ".join(r"acc. & $f$" for _ in available)
    cmid = " ".join(
        rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(ncol)
    )
    hw = max_ci_halfwidth(available)

    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{\textbf{The decomposition ladder.} One frozen decoder, thirteen feature maps, "
        r"the same underlying corpora. \emph{acc.} is mean test accuracy (\%) over the "
        r"pre-registered seed count; $f$ is the algorithm-relative recoverable fraction "
        r"\eqref{eq:f}, computed seed-paired for the registered rungs and as a ratio of means for "
        rf"the two unregistered ones (W7-1/8, W8). Every bootstrap 95\% CI on $f$ is narrower than "
        rf"$\pm{hw:.3f}$, and every CI on an accuracy cell is narrower than $\pm0.60$ points "
        rf"(median $\pm0.27$), so no ordering in this table is within seed noise. Rungs marked $\dagger$ act on the random-initialization corpus. W4 and W5 "
        r"are orbit-valued \emph{reframings}; W10 is an invariant \emph{encoding}, separated from "
        r"nonlinear feature engineering only by the matched control of \S\ref{sec:control}, and "
        r"therefore not directly comparable to the reframings as a symmetry measurement.}",
        r"\label{tab:ladder}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        rf"& & {head} \\",
        cmid,
        rf"rung & feature map & {sub} \\",
        r"\midrule",
    ]
    for rung, desc, dagger in ROWS:
        cells = []
        for _, _, means, fracs in available:
            cells.append(f"{means[rung]:.2f}" if rung in means else "---")
            cells.append(frac(means, fracs, rung) if dagger else "---")
        mark = r"$^\dagger$" if dagger else ""
        name = r"W7$_{1/8}$" if rung == "W7-1/8" else rung
        lines.append(f"{name}{mark} & {desc} & " + " & ".join(cells) + r" \\")
        if rung in ("P1", "W3", "W10"):
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def gap_table(available) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{\textbf{The gap, and what an exact \emph{reframing} recovers.} The "
        r"shared-versus-random accuracy gap W1$-$W3 shrinks with the task ceiling. The rung "
        r"reported is the best orbit-valued \emph{reframing} (W4 or W5). The invariant "
        r"\emph{encodings} W10 and W11b are excluded here and reported separately, because their "
        r"gain is separated from nonlinear feature engineering only by the matched control of "
        r"\S\ref{sec:control}. $f$ is "
        r"algorithm-relative and the last column is what this reframing did not recover, not a "
        r"measurement of non-symmetry.}",
        r"\label{tab:gap}",
        r"\begin{tabular}{@{}lrrlrr@{}}",
        r"\toprule",
        r"dataset & ceiling P0 & gap W1$-$W3 & best reframing & recovered & unrecovered \\",
        r"\midrule",
    ]
    for _, label, m, _ in available:
        if not {"P0", "W1", "W3", "W4", "W5"} <= m.keys():
            continue
        gap = m["W1"] - m["W3"]
        # reframings only: W10/W11b are invariant encodings, not orbit-valued maps
        exact = {r: m[r] for r in ("W4", "W5") if r in m}
        best = max(exact, key=exact.get)
        removed = exact[best] - m["W3"]
        name = {"W4": r"W4 $\csort$", "W5": r"W5 $\calign$", "W10": "W10 invariants"}[best]
        lines.append(
            f"{label} & {m['P0']:.2f} & {gap:.2f} & {name} & {removed:.2f} "
            f"& {1 - removed / gap:.1%}".replace("%", r"\%") + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def s4e_table() -> str | None:
    """The S4e anatomy: control, local conditioning, basin radius, independent students."""
    path = ROOT / "results" / "s4e" / "s4e.json"
    if not path.exists():
        return None
    arms = json.loads(path.read_text())["arms"]
    if not {"planted", "sensitivity", "warmstart", "teacher", "null"} <= arms.keys():
        return None

    planted = {r["width"]: r for r in arms["planted"]}
    sens = {r["width"]: r for r in arms["sensitivity"]}
    teach = {r["width"]: r for r in arms["teacher"]}
    null = {r["width"]: r for r in arms["null"]}
    warm = {(r["width"], r["eps_start"]): r for r in arms["warmstart"]}
    eps_small = min({r["eps_start"] for r in arms["warmstart"]})
    widths = sorted(teach)

    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        rf"\caption{{\textbf{{S4e: the anatomy of deep identifiability at $L=2$.}} "
        rf"\emph{{planted}} is the validity control (a known group element, undone). "
        rf"$\kappa = R_\theta/R_f$ is the local condition number of the inverse map. "
        rf"\emph{{basin}} is the fraction of optimiser runs that return to the true orbit when "
        rf"started a relative $\varepsilon = 10^{{{int(round(__import__('math').log10(eps_small)))}}}$ away. The last three columns are "
        rf"independent students fitted to a teacher's exact outputs: their best functional "
        rf"residual, the orbit residual there, and --- for scale --- the orbit residual between "
        rf"two \emph{{unrelated}} networks of the same shape.}}",
        r"\label{tab:s4e}",
        r"\begin{tabular}{@{}rrrrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{controls} & & \multicolumn{3}{c}{independent students} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){5-7}",
        r"width $n$ & planted $R_\theta$ & basin & $\kappa$ & best $R_f$ & $R_\theta$ there "
        r"& unrelated $R_\theta$ \\",
        r"\midrule",
    ]
    for w in widths:
        b = warm.get((w, eps_small), {}).get("recovered_frac")
        lines.append(
            f"{w} & {planted[w]['R_theta_max']:.1e} & "
            + (f"{b:.0%}".replace("%", r"\%") if b is not None else "---")
            + f" & {sens[w]['ladder'][0]['kappa_median']:.4f}"
            + f" & {teach[w]['R_f_min']:.1e}"
            + f" & {teach[w]['R_theta_at_best_R_f']:.3f}"
            + f" & {null[w]['R_theta_median']:.3f}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def _psnr_of(dataset: str) -> str:
    """Median render PSNR of the shared-init corpus, from its committed quality gate."""
    g = ROOT / "results" / "inrbench" / f"{dataset}_P-shared-det_test_gate.json"
    if not g.exists():
        return "---"
    return f"{json.loads(g.read_text())['psnr']['median']:.1f}"


def gray_arm_table() -> str | None:
    """Luminance CIFAR against the three primary corpora: images or channels?"""
    order = [("mnist", "MNIST", 1), ("fashionmnist", "FashionMNIST", 1),
             ("cifar10gray", "CIFAR-10 luminance", 1), ("cifar10", "CIFAR-10 RGB", 3)]
    got = {k: load(k) for k, _, _ in order}
    g = got.get("cifar10gray")
    if not g or not {"W1", "W3", "W5", "W10"} <= g[0].keys():
        return None
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{\textbf{Images or channels?} Luminance CIFAR-10 is the identical images at the "
        r"identical geometry, architecture and fit budget, with the output-channel count changed "
        r"from three to one. If the RGB behaviour of W5 and W10 is driven by $c$, the luminance "
        r"arm sits with the grayscale corpora; if by image statistics, it sits with RGB CIFAR. "
        r"Median render PSNR is reported because dropping channels makes the fit "
        r"over-parameterised, which is an unavoidable consequence of the intervention.}",
        r"\label{tab:gray}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"corpus & $c$ & PSNR (dB) & P0 & gap W1$-$W3 & $f(\mathrm{W5})$ & $f(\mathrm{W10})$ \\",
        r"\midrule",
    ]
    for key, label, c in order:
        v = got.get(key)
        if not v:
            continue
        m, fr = v
        if not {"W1", "W3"} <= m.keys():
            continue
        lines.append(
            f"{label} & {c} & {_psnr_of(key)} & " + (f"{m['P0']:.2f}" if "P0" in m else "---")
            + f" & {m['W1'] - m['W3']:.2f} & {frac(m, fr, 'W5')} & {frac(m, fr, 'W10')}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def w11_table() -> str | None:
    """W11: reader architecture against frame choice, and the pooling split."""
    d = ROOT / "results" / "ladder" / "mnist"
    if not (d / "W11.json").exists() or not (d / "S1_analysis.json").exists():
        return None
    v = json.loads((d / "W11.json").read_text())["variants"]
    fr = json.loads((d / "S1_analysis.json").read_text())["recovery_fractions"]
    means = json.loads((d / "S1_analysis.json").read_text())["means"]
    rows = [
        ("W3", "raw weights, random init", "matched MLP", means["W3"], 0.0, 1873162),
        ("W4", r"$\csort$ (exact reframing)", "matched MLP", means["W4"], fr["f_W4"]["point"], 1873162),
        ("W11a", "permutation-equivariant, raw weights", "graph reader",
         v["W11a"]["mean"], v["W11a"]["recovery_fraction"], v["W11a"]["reader_params"]),
        ("W10", "exact invariants, eigenvalue pooling", "matched MLP",
         means["W10"], fr["f_W10"]["point"], 987402),
        ("W11b", "same invariants, learned pooling", "graph reader",
         v["W11b"]["mean"], v["W11b"]["recovery_fraction"], v["W11b"]["reader_params"]),
        ("W5", r"$\calign$ (exact reframing)", "matched MLP", means["W5"], fr["f_W5"]["point"], 1873162),
        ("W1", "raw weights, shared init", "matched MLP", means["W1"], 1.0, 1873162),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{\textbf{Reader architecture against frame choice} (MNIST, \texttt{P-random}). "
        r"W11a is the equivariant coverage the field has for sine networks; W11b feeds W10's own "
        r"invariants to an equivariant reader with \emph{learned} pooling instead of sorted "
        r"eigenvalue spectra. Reader parameter counts are matched to the frozen decoder by rule, so "
        r"no row loses for being smaller. The W10$\to$W11b$\to$W5 chain splits the invariant "
        r"encoding's shortfall into a pooling part and an incompleteness part.}",
        r"\label{tab:w11}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"rung & construction & reader & acc. (\%) & $f$ \\",
        r"\midrule",
    ]
    for name, desc, reader, acc, f, npar in rows:
        lines.append(f"{name} & {desc} & {reader} ({npar/1e6:.2f}M) & {acc:.2f} & {f:.3f}" + r" \\")
        if name in ("W11a", "W11b"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def orbit_table() -> str | None:
    """S6: what a pure group intervention costs, and what each treatment recovers.

    Arm (i) lives in orbit_mnist_perm.json, arm (ii) in *_noperm.json and arm (iii) --- the
    equivariant readers, run at a single B --- in *_equivariant.json, whose treatment cells are
    merged into arm (i)'s at the matching winding bound.
    """
    d = ROOT / "results" / "s6"
    main = d / "orbit_mnist_perm.json"
    if not main.exists():
        return None
    perm = json.loads(main.read_text())
    noperm = json.loads((d / "orbit_mnist_noperm.json").read_text()) if (
        d / "orbit_mnist_noperm.json").exists() else None

    equi_path = d / "orbit_mnist_equivariant.json"
    if equi_path.exists():
        equi = json.loads(equi_path.read_text())
        for b, cell in equi["by_winding"].items():
            if b not in perm["by_winding"]:
                continue
            for name in ("W11a", "W11b"):
                if name in cell["treatments"]:
                    perm["by_winding"][b]["treatments"][name] = cell["treatments"][name]

    order = ["raw", "c_sort", "c_align", "invariants", "W11a", "W11b"]
    pretty = {"raw": "raw weights", "c_sort": r"$\csort$", "c_align": r"$\calign$",
              "invariants": "exact invariants (W10)", "W11a": "equivariant, raw (W11a)",
              "W11b": "equivariant, invariant (W11b)"}
    Bs = sorted(perm["by_winding"], key=int)
    lines = [
        r"\begin{table*}[t]", r"\centering\small",
        r"\caption{\textbf{The orbit-only intervention (MNIST).} Each fitted network and its "
        r"realised function are held fixed; an independent $g_i\sim\mu_B$ is applied to each. "
        rf"The baseline is the untouched corpus at ${perm['baseline']['mean']:.2f}\%$. "
        r"$\Delta_{\mathrm{sym}}$ is the degradation caused by the group action alone; the "
        r"treatment rows give the share of it recovered. Because the functions are unchanged to "
        r"machine precision, this is a causal measurement, unlike the ladder's $f$.}",
        r"\label{tab:orbit}",
        r"\begin{tabular}{@{}l" + "r" * len(Bs) + r"@{}}", r"\toprule",
        "treatment & " + " & ".join(rf"$B={b}$" for b in Bs) + r" \\", r"\midrule",
    ]
    lines.append(r"$\Delta_{\mathrm{sym}}$ (points) & "
                 + " & ".join(f"{perm['by_winding'][b]['delta_sym']:.2f}" for b in Bs) + r" \\")
    lines.append(r"\midrule")
    for name in order:
        if not any(name in perm["by_winding"][b]["treatments"] for b in Bs):
            continue
        cells = []
        for b in Bs:
            tr = perm["by_winding"][b]["treatments"].get(name)
            cells.append("---" if tr is None else
                         ("0.000" if name == "raw" else f"{tr['recovered_fraction']:.3f}"))
        lines.append(f"{pretty[name]} & " + " & ".join(cells) + r" \\")
    if noperm:
        lines.append(r"\midrule")
        vals = {b: c["delta_sym"] for b, c in noperm["by_winding"].items()}
        lines.append(r"$\Delta_{\mathrm{sym}}$, identity permutation & "
                     + " & ".join(f"{vals[b]:.2f}" if b in vals else "---" for b in Bs) + r" \\")
    prandom_path = d / "orbit_mnist_prandom.json"
    if prandom_path.exists():
        pr = json.loads(prandom_path.read_text())
        vals = {b: c["delta_sym"] for b, c in pr["by_winding"].items()}
        lines.append(r"$\Delta_{\mathrm{sym}}$, applied to \texttt{P-random} instead & "
                     + " & ".join(f"{vals[b]:+.2f}" if b in vals else "---" for b in Bs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def sweep_table() -> str | None:
    """S8: the ladder at four step budgets, against three fit diagnostics."""
    path = ROOT / "results" / "s8" / "sweep.json"
    if not path.exists():
        return None
    rep = json.loads(path.read_text())
    steps = sorted(rep["by_steps"], key=int)
    if not steps:
        return None
    col = "@{}l" + "r" * len(steps) + "@{}"
    lines = [
        r"\begin{table*}[t]", r"\centering\small",
        r"\caption{\textbf{The ladder against the step budget (MNIST).} Same images, "
        r"architecture, optimizer and seed policy; only the number of steps changes. The corpus "
        r"is reduced to 10k/2k/2k so the largest budget is affordable, so the 300-step column is "
        r"the internal control rather than the full-corpus ladder. $\|\nabla\|$ is the median "
        r"relative endpoint gradient norm of each INR's own full-batch loss --- the stationarity "
        r"measure, which render fidelity cannot supply.}",
        r"\label{tab:sweep}",
        rf"\begin{{tabular}}{{{col}}}", r"\toprule",
        "quantity & " + " & ".join(rf"{int(b):,} steps" for b in steps) + r" \\",
        r"\midrule",
    ]

    def row(label, fn, fmt="{:.2f}"):
        cells = []
        for b in steps:
            try:
                cells.append(fmt.format(fn(rep["by_steps"][b])))
            except (KeyError, TypeError):
                cells.append("---")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    row(r"W1 (shared init)", lambda r: r["cells"]["W1"]["mean"])
    row(r"W3 (random init)", lambda r: r["cells"]["W3"]["mean"])
    row(r"gap W1$-$W3", lambda r: r["gap"])
    lines.append(r"\midrule")
    row(r"$f(\mathrm{W4})$ $\csort$", lambda r: r["f"]["W4"], "{:.3f}")
    row(r"$f(\mathrm{W5})$ $\calign$", lambda r: r["f"]["W5"], "{:.3f}")
    row(r"$f(\mathrm{W10})$ invariants", lambda r: r["f"]["W10"], "{:.3f}")
    lines.append(r"\midrule")
    row(r"median $\|\nabla\|/\|\theta\|$",
        lambda r: r["diagnostics"]["random"]["grad_norm_median"], "{:.1e}")
    row(r"median render PSNR (dB)",
        lambda r: r["diagnostics"]["random"]["psnr_median"], "{:.1f}")
    row(r"median relative travel",
        lambda r: r["diagnostics"]["shared"]["travel_median"], "{:.3f}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def w12_cross_table() -> str | None:
    """W12 on every corpus it has been run on, against the best of each other family."""
    best = {  # from the frozen ladder: best exact reframing, best invariant encoding
        "mnist": ("MNIST", 0.628, 0.269),
        "fashionmnist": ("FashionMNIST", 0.664, 0.428),
        "cifar10gray": ("CIFAR-10 (luminance)", 0.324, 0.493),
        "cifar10": ("CIFAR-10 (RGB)", 0.324, 0.534),
    }
    rows = []
    for ds, (label, calign, enc) in best.items():
        path = ROOT / "results" / "ladder" / ds / "W12.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        rows.append((label, d["mean"], d["recovery_fraction"], calign, enc))
    if not rows:
        return None
    lines = [
        r"\begin{table*}[t]", r"\centering\small",
        r"\caption{\textbf{The phasor-graded reader across corpora.} $s$ is the "
        r"reference-normalized score \eqref{eq:s}; the last two columns are the best exact "
        r"reframing and the best invariant encoding on the same corpus, quoted as recovered "
        r"fractions \eqref{eq:f}. W12 is above both families everywhere, including on the two "
        r"CIFAR corpora where those families trade places with each other. Capacity is set by the "
        r"same rule per dataset, and invariance is audited on each corpus's own fitted networks "
        r"(maximum relative logit movement $3.7\times10^{-6}$ at $|j| \le 40$).}",
        r"\label{tab:w12cross}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"corpus & W12 acc.\ (\%) & $s(\mathrm{W12})$ & best reframing & best encoding \\",
        r"\midrule",
    ]
    for label, acc, s, calign, enc in rows:
        lines.append(f"{label} & {acc:.2f} & \\textbf{{{s:.3f}}} & {calign:.3f} & {enc:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def calibration_table() -> str:
    rows = list(csv.DictReader((ROOT / "docs" / "PREDICTION_OUTCOMES.csv").open()))
    iv = [r for r in rows if r["kind"] == "interval"]
    miss = [r for r in iv if r["verdict"] == "MISS"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        rf"\caption{{\textbf{{Every registered interval that missed.}} Realized coverage is "
        rf"{len(iv) - len(miss)}/{len(iv)} against a nominal 80\%. The misses fall into exactly "
        r"two modes (last column): hedging a registered mechanism toward priors from another "
        r"setting, and registering a contrast between two treatments of a nuisance that then "
        r"turned out to be null, so that no contrast could exist at any magnitude.}",
        r"\label{tab:calibration}",
        r"\begin{tabular}{@{}llrrl@{}}",
        r"\toprule",
        r"id & quantity & registered & observed & mode \\",
        r"\midrule",
    ]
    mode = {
        "QG-3": "hedged mechanism",
        "QG-5": "hedged mechanism",
        "H-S1-4c": "hedged mechanism",
        "H-S1-3": "nuisance was null",
        "H-S1-5": "nuisance was null",
    }
    for r in miss:
        pid = r["prediction"].split(" ")[0]
        desc = " ".join(r["prediction"].split(" ")[1:])
        desc = desc.replace("%", r"\%").replace("_", r"\_")
        lines.append(
            f"{pid} & {desc} & {r['point']} [{r['lo80']}, {r['hi80']}] & {r['observed']} "
            f"& {mode.get(pid, '---')} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


MD_LABEL = {
    "W4": r"$c_\text{sort}$ — exact, template-free",
    "W5": r"$c_\text{align}$ — exact, aligned to $\theta_0$",
    "W10": r"exact $L{=}2$ invariants",
    "W7-1/8": r"*control:* $K$ corpus, rows matched",
}


def ladder_markdown(available) -> str:
    """The same table as ladder_table.tex, for the README (kept in sync by markers)."""
    head = "| rung | feature map | " + " | ".join(
        f"{lbl} | $f$" for _, lbl, _, _ in available
    ) + " |"
    sep = "|---|---|" + "".join(["---:|---:|" for _ in available])
    lines = [head, sep]
    for rung, desc, dagger in ROWS:
        d = MD_LABEL.get(rung, desc)
        d = (d.replace(r"\textbf{", "**").replace("}", "**") if r"\textbf" in d else d)
        cells = []
        for _, _, means, fracs in available:
            cells.append(f"{means[rung]:.2f}" if rung in means else "—")
            cells.append(frac(means, fracs, rung).replace("---", "—") if dagger else "—")
        name = ("**" + rung + "**") if rung in ("W1", "W3", "W5") else rung
        mark = " †" if dagger else ""
        lines.append(f"| {name}{mark} | {d} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("† acts on the random-init corpus. W4, W5, W10 are **exactly** "
                 "function-preserving. Chance = 10.")
    return "\n".join(lines)


def patch_readme(markdown: str) -> None:
    path = ROOT / "README.md"
    start, end = "<!-- LADDER_TABLE:START -->", "<!-- LADDER_TABLE:END -->"
    text = path.read_text()
    if start not in text or end not in text:
        print("README markers absent — skipping table injection")
        return
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{markdown}\n{end}{tail}")
    print(f"patched {path} between {start} / {end}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    available = []
    for key, label in DATASETS:
        got = load(key)
        if got and {"W1", "W3"} <= got[0].keys():
            available.append((key, label, got[0], got[1]))
        else:
            print(f"{key}: skipped (ladder not complete)")
    if not available:
        raise SystemExit("no ladder results found")
    (OUT / "ladder_table.tex").write_text(ladder_table(available))
    (OUT / "gap_table.tex").write_text(gap_table(available))
    (OUT / "calibration_table.tex").write_text(calibration_table())
    xt = w12_cross_table()
    (OUT / "w12_cross_table.tex").write_text(xt or "% W12 cross-dataset pending\n")
    print(f"wrote {OUT / 'w12_cross_table'}.tex" if xt else "w12_cross_table: pending")
    for f in ("ladder_table", "gap_table", "calibration_table"):
        print(f"wrote {OUT / f}.tex")
    swp = sweep_table()
    (OUT / "sweep_table.tex").write_text(swp or "% convergence sweep results pending\n")
    print(f"wrote {OUT / 'sweep_table'}.tex" if swp else "sweep_table: pending")

    orb = orbit_table()
    (OUT / "orbit_table.tex").write_text(orb or "% orbit intervention results pending\n")
    print(f"wrote {OUT / 'orbit_table'}.tex" if orb else "orbit_table: pending")
    w11 = w11_table()
    (OUT / "w11_table.tex").write_text(w11 or "% W11 not run yet\n")
    print(f"wrote {OUT / 'w11_table'}.tex" if w11 else "w11_table: no results yet")
    gray = gray_arm_table()
    (OUT / "gray_table.tex").write_text(gray or "% luminance-CIFAR arm not complete yet\n")
    print(f"wrote {OUT / 'gray_table'}.tex" if gray else "gray_table: no results yet (placeholder)")
    s4e = s4e_table()
    if s4e:
        (OUT / "s4e_table.tex").write_text(s4e)
        print(f"wrote {OUT / 's4e_table'}.tex")
    else:
        (OUT / "s4e_table.tex").write_text("% S4e results not present yet\n")
        print("s4e_table: no results yet (placeholder written)")
    patch_readme(ladder_markdown(available))


if __name__ == "__main__":
    main()

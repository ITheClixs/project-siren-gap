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
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{\textbf{The decomposition ladder.} One frozen decoder, thirteen feature maps, "
        r"the same underlying corpora. \emph{acc.} is mean test accuracy (\%) over the "
        r"pre-registered seed count; $f$ is the recovery fraction \eqref{eq:f}, computed "
        r"seed-paired for the registered rungs and as a ratio of means for the two unregistered "
        rf"ones (W7-1/8, W8). Every bootstrap 95\% CI on $f$ is narrower than $\pm{hw:.3f}$. "
        r"Rungs marked $\dagger$ act on the random-initialization corpus and are attempts to "
        r"close the W3$\to$W1 gap; W4, W5 and W10 are \emph{exactly} function-preserving.}",
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
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def gap_table(available) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{\textbf{The gap, and what an exact treatment of the group removes.} The "
        r"perception gap W1$-$W3 shrinks in absolute terms with the task ceiling. The best "
        r"\emph{exact}, function-preserving rung (W4, W5 or W10 --- the winner changes) is a "
        r"certified lower bound on the symmetry-attributable share, so the last column is an "
        r"\emph{upper} bound on what is not symmetry, not a measurement of it "
        r"(Proposition~\ref{prop:bound}).}",
        r"\label{tab:gap}",
        r"\begin{tabular}{@{}lrrlrr@{}}",
        r"\toprule",
        r"dataset & ceiling P0 & gap W1$-$W3 & best exact rung & removed & residual $\le$ \\",
        r"\midrule",
    ]
    for _, label, m, _ in available:
        if not {"P0", "W1", "W3", "W4", "W5"} <= m.keys():
            continue
        gap = m["W1"] - m["W3"]
        exact = {r: m[r] for r in ("W4", "W5", "W10") if r in m}
        best = max(exact, key=exact.get)
        removed = exact[best] - m["W3"]
        name = {"W4": r"W4 $\csort$", "W5": r"W5 $\calign$", "W10": "W10 invariants"}[best]
        lines.append(
            f"{label} & {m['P0']:.2f} & {gap:.2f} & {name} & {removed:.2f} "
            f"& {1 - removed / gap:.1%}".replace("%", r"\%") + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def calibration_table() -> str:
    rows = list(csv.DictReader((ROOT / "docs" / "PREDICTION_OUTCOMES.csv").open()))
    iv = [r for r in rows if r["kind"] == "interval"]
    miss = [r for r in iv if r["verdict"] == "MISS"]
    lines = [
        r"\begin{table}[t]",
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
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
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
    for f in ("ladder_table", "gap_table", "calibration_table"):
        print(f"wrote {OUT / f}.tex")
    patch_readme(ladder_markdown(available))


if __name__ == "__main__":
    main()

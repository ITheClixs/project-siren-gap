#!/usr/bin/env python3
"""Regenerate paper/tables/bench_table.tex from the S13 artifacts.

Baseline accuracies are quoted from the ScaleGMN paper's Table 1 and are constants here; our own
row is read from the three W12_dwsbench artifacts, so the table cannot drift from the runs that
produced it.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = [
    (r"ScaleGMN-B \citep{kalogeropoulos2024scalegmn}", (96.59, 80.78, 38.82)),
    (r"ScaleGMN \citep{kalogeropoulos2024scalegmn}", (96.57, 80.46, 36.43)),
    (None, None),
    (r"NG-GNN \citep{kofinas2024graph}", (91.40, 68.00, 36.04)),
    (r"DWSNets \citep{navon2023dws}", (85.71, 67.06, 34.45)),
    (r"NFN$_{HNP}$ \citep{zhou2023nfn}", (79.11, 68.94, 28.64)),
    (r"NFN$_{NP}$ \citep{zhou2023nfn}", (78.50, 68.19, 33.41)),
    # Reported in its own paper under its own training protocol (its NFN_NP row reads 69.82 on
    # MNIST there, against 78.50 in ScaleGMN's Table 1), so this row is not cell-comparable.
    (r"Monomial-NFN$^\ddagger$ \citep{tran2024monomial}", (68.43, 61.15, 34.23)),
]
# Readers that also evaluate the network at input coordinates. Quoted, not rerun: NG-GNN and NG-T
# with 64 probe features as reported in the ScaleGMN paper's Section 5 text, ProbeGen with 128
# probes from its Table 3. None of them reports CIFAR-10 INR classification.
QUERYING = [
    (r"NG-GNN, 64 probes \citep{kofinas2024graph}", (94.70, 74.20, None)),
    (r"NG-T, 64 probes \citep{kofinas2024graph}", (97.30, 74.80, None)),
    (r"ProbeGen, 128 probes \citep{kahana2025probegen}", (98.40, 87.70, None)),
]
CAPTION = (
    r"\caption{\textbf{The phasor-graded reader against the published weight-space leaderboard.} "
    r"Test accuracy (\%) on the standard INR-classification corpora, MNIST and FashionMNIST "
    r"released by \citet{navon2023dws} and CIFAR-10 by \citet{zhou2023nfn}, whose networks have "
    r"the architecture our readers already take, so W12 runs on them unchanged and in its frozen "
    r"configuration with nothing tuned per dataset, after absorbing SIREN's $\omega_0$ into the "
    r"stored weights so that they are in the canonical form of \S\ref{sec:group}. Baselines are "
    r"quoted from ScaleGMN's Table~1 except $^\ddagger$, quoted from its own paper under its own "
    r"protocol; ours are mean $\pm$ population s.d.\ over five seeds. Among readers of weights alone W12 places third, third and second. The lower "
    r"block also evaluates each network at learned input coordinates, which is function access in "
    r"the sense of \S\ref{sec:function}, and is stronger still; none of those methods reports "
    r"CIFAR-10. Inr2Array \citep{zhou2023nft} reports $98.5$, $79.3$ and $63.4$ on corpora rebuilt "
    r"with the procedure of \citet{zhou2023nfn}, which are different corpora from these and are "
    r"not listed. The FashionMNIST cell uses the authors' own split file; for MNIST and CIFAR-10 no "
    r"split file is distributed, so validation is carved deterministically from the training half "
    r"and the test half is theirs. All three corpora are independently initialized by the measure "
    r"of \S\ref{sec:orbit}.}"
)


def fmt(v: float | None) -> str:
    return "--" if v is None else f"{v:.2f}"


def main() -> None:
    acc, sd = [], []
    for ds in ("mnist", "fashionmnist", "cifar10"):
        cell = ROOT / "results" / "ladder" / ds / "W12_dwsbench_w0.json"
        if not cell.exists():
            raise SystemExit(f"missing {cell}: run scripts/74_bench_w0_chain.sh first")
        d = json.loads(cell.read_text())
        acc.append(d["mean"])
        sd.append(statistics.pstdev(d["acc"]))
    lines = [r"\begin{table*}[t]", r"\centering\small", CAPTION, r"\label{tab:bench}",
             r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"method & MNIST & FashionMNIST & CIFAR-10 \\", r"\midrule"]
    for name, vals in PUBLISHED:
        if name is None:
            lines.append(r"\textbf{W12 (ours)} & "
                         + " & ".join(rf"\textbf{{{a:.2f}}} $\pm$ {s:.2f}"
                                      for a, s in zip(acc, sd)) + r" \\")
        else:
            lines.append(f"{name} & " + " & ".join(fmt(v) for v in vals) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{@{}l}{\emph{also querying the network}} \\")
    for name, vals in QUERYING:
        lines.append(f"{name} & " + " & ".join(fmt(v) for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out = ROOT / "paper" / "tables" / "bench_table.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out}: " + ", ".join(f"{a:.2f}" for a in acc))


if __name__ == "__main__":
    main()

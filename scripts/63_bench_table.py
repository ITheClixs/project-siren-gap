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
]
CAPTION = (
    r"\caption{\textbf{The phasor-graded reader against the published weight-space leaderboard.} "
    r"Test accuracy (\%) on the standard INR-classification corpora --- MNIST and FashionMNIST "
    r"released by \citet{navon2023dws}, CIFAR-10 by \citet{zhou2023nfn} --- whose networks have the "
    r"architecture our readers already take, so W12 runs on them unchanged and in its frozen "
    r"configuration with nothing tuned per dataset. Baselines are quoted from the ScaleGMN "
    r"paper's Table~1; ours are five seeds. W12 places third, third and second, above DWSNets, "
    r"both NFN variants and NG-GNN everywhere, and above ScaleGMN itself on CIFAR-10. The "
    r"FashionMNIST cell uses the authors' own split file; for MNIST and CIFAR-10 no split file "
    r"is distributed, so validation is carved deterministically from the training half and the "
    r"test half is theirs. All three corpora are independently initialized by the measure of "
    r"\S\ref{sec:orbit}.}"
)


def main() -> None:
    acc, sd = [], []
    for ds in ("mnist", "fashionmnist", "cifar10"):
        d = json.loads((ROOT / "results" / "ladder" / ds / "W12_dwsbench.json").read_text())
        acc.append(d["mean"])
        sd.append(statistics.pstdev(d["acc"]))
    lines = [r"\begin{table}[t]", r"\centering\small", CAPTION, r"\label{tab:bench}",
             r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"method & MNIST & FashionMNIST & CIFAR-10 \\", r"\midrule"]
    for name, vals in PUBLISHED:
        if name is None:
            lines.append(r"\textbf{W12 (ours)} & "
                         + " & ".join(rf"\textbf{{{a:.2f}}} $\pm$ {s:.2f}"
                                      for a, s in zip(acc, sd)) + r" \\")
        else:
            lines.append(f"{name} & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = ROOT / "paper" / "tables" / "bench_table.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out}: " + ", ".join(f"{a:.2f}" for a in acc))


if __name__ == "__main__":
    main()

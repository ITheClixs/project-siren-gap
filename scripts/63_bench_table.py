#!/usr/bin/env python3
"""Regenerate paper/tables/bench_table.tex from the S13 artifact and the published numbers.

Baseline accuracies are quoted from the ScaleGMN paper's Table 1 and are constants here; only
our own row is read from an artifact, so the table cannot drift from the run that produced it.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = [
    ("ScaleGMN-B \\citep{kalogeropoulos2024scalegmn}", 96.59, 0.24),
    ("ScaleGMN \\citep{kalogeropoulos2024scalegmn}", 96.57, 0.10),
    (None, None, None),  # our row is spliced in by accuracy order
    ("NG-GNN \\citep{kofinas2024graph}", 91.40, 0.60),
    ("DWSNets \\citep{navon2023dws}", 85.71, 0.57),
    ("NFN$_{HNP}$ \\citep{zhou2023nfn}", 79.11, 0.84),
    ("NFN$_{NP}$ \\citep{zhou2023nfn}", 78.50, 0.23),
]
CAPTION = (
    r"\caption{\textbf{The phasor-graded reader on the standard MNIST-INR benchmark.} Test "
    r"accuracy (\%) on the corpus of \citet{navon2023dws}, which the published weight-space "
    r"literature reports on and whose INRs have the architecture our readers already take. "
    r"Baseline numbers are quoted from the ScaleGMN paper's Table~1. W12 is run in its frozen "
    r"configuration with no tuning against this benchmark, at five seeds. It is third of seven: "
    r"ahead of DWSNets, both NFN variants and NG-GNN, and behind ScaleGMN. The official split "
    r"file is not in the distributed archive, so validation is carved deterministically from the "
    r"training half (54k/6k/10k), which is our split and not theirs.}"
)


def main() -> None:
    d = json.loads((ROOT / "results" / "ladder" / "mnist" / "W12_dwsbench.json").read_text())
    acc, sd = d["mean"], statistics.pstdev(d["acc"])
    lines = [r"\begin{table}[t]", r"\centering\small", CAPTION, r"\label{tab:bench}",
             r"\begin{tabular}{@{}lr@{}}", r"\toprule",
             r"method & MNIST-INR test accuracy (\%) \\", r"\midrule"]
    for name, val, err in PUBLISHED:
        if name is None:
            lines.append(f"\\textbf{{W12 (ours)}} & {acc:.2f} $\\pm$ {sd:.2f} " + r"\\")
        else:
            lines.append(f"{name} & {val:.2f} $\\pm$ {err:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = ROOT / "paper" / "tables" / "bench_table.tex"
    out.write_text("\n".join(lines))
    print(f"wrote {out} (W12 {acc:.2f} +/- {sd:.2f})")


if __name__ == "__main__":
    main()

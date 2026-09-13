#!/usr/bin/env python3
"""S17: replicate the shared initialization theta_0 (docs/prereg/S17.md).

Fits two new P-shared-det corpora at S8's reduced size and budget with different shared inits
(--seed-offset 1, 2), then decodes, for each of the three draws, W1, W5 (c_align of the shared
P-random-s8s300 corpus to that draw's theta_0) and the S6 intervention at B = 0.

    .venv/bin/python scripts/77_s17_theta0.py --dataset mnist
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sirengap.eval import rungs  # noqa: E402
from sirengap.eval.decoder import train_matched_mlp  # noqa: E402
from sirengap.eval.rungs import CorpusCache, _chunked, probe_coords  # noqa: E402
from sirengap.eval.stats import bootstrap_ci_mean  # noqa: E402
from sirengap.canon.calign import c_align  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402

S6 = __import__("37_orbit_intervention")

SEEDS = 5
DRAWS = {0: "P-shared-det-s8s300", 1: "P-shared-det-s17o1", 2: "P-shared-det-s17o2"}
RANDOM = "P-random-s8s300"


def fit_missing(dataset: str, root: Path) -> None:
    for offset, protocol in DRAWS.items():
        if offset == 0 or (root / dataset / protocol).exists():
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "03_generate_inrbench.py"),
               "--dataset", dataset, "--protocol", "P-shared-det", "--steps", "300",
               "--n-train", "10000", "--n-val", "2000", "--n-test", "2000",
               "--tag", f"s17o{offset}", "--seed-offset", str(offset), "--out-root", str(root)]
        print("fitting:", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True)


def template_for(params: SirenParams, offset: int) -> SirenParams:
    widths = params.widths()
    return rungs.absorb_omega(rungs.init_from_seeds([offset], params.hidden[0][0].shape[2],
                                                    widths, params.w_out.shape[1]))


def decode(feats: dict, labels: dict, seeds: int, device: str) -> dict:
    a = np.array([train_matched_mlp(feats, labels, seed=s, device=device).test_acc
                  for s in range(seeds)])
    return {"acc": a.tolist(), "mean": float(a.mean()), "ci95": bootstrap_ci_mean(a)}


def median_travel(params: SirenParams, template: SirenParams) -> float:
    t = template.flat()[0]
    d = (params.flat() - t).norm(dim=1) / t.norm()
    return float(d.median())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--root", default="data/inrbench")
    args = ap.parse_args()
    root = Path(args.root)

    fit_missing(args.dataset, root)
    cache = CorpusCache(root / args.dataset, args.dataset)
    probes = probe_coords(args.dataset)
    rnd, rnd_labels = cache.split_params(RANDOM)
    w3 = decode({s: p.flat() for s, p in rnd.items()}, rnd_labels, args.seeds, args.device)
    out: dict = {"study": "S17 theta_0 replicates", "prereg": "docs/prereg/S17.md",
                 "dataset": args.dataset, "seeds": args.seeds, "W3": w3, "draws": {}}
    print(f"W3 {w3['mean']:.2f}", flush=True)

    for offset, protocol in DRAWS.items():
        t0 = time.time()
        shared, labels = cache.split_params(protocol)
        template = template_for(shared["train"], offset)
        w1 = decode({s: p.flat() for s, p in shared.items()}, labels, args.seeds, args.device)
        aligned = {s: _chunked(lambda q: c_align(q, template, probes)[0].flat(), p)
                   for s, p in rnd.items()}
        w5 = decode(aligned, rnd_labels, args.seeds, args.device)
        scattered = S6.scatter_corpus(shared, 0, seed=1234, permute=True)
        sc = decode({s: p.flat() for s, p in scattered.items()}, labels, args.seeds, args.device)
        f_w5 = (w5["mean"] - w3["mean"]) / (w1["mean"] - w3["mean"])
        out["draws"][str(offset)] = {
            "protocol": protocol, "W1": w1, "W5": w5, "f_W5": f_w5,
            "delta_sym_B0": w1["mean"] - sc["mean"], "scattered": sc,
            "median_relative_travel": median_travel(shared["train"], template),
            "wallclock_s": time.time() - t0}
        d = out["draws"][str(offset)]
        print(f"draw {offset}: W1 {w1['mean']:.2f}  W5 {w5['mean']:.2f}  f(W5) {f_w5:.3f}  "
              f"delta_sym {d['delta_sym_B0']:.2f}  travel {d['median_relative_travel']:.3f}",
              flush=True)

    path = ROOT / "results" / "s17" / f"theta0_{args.dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

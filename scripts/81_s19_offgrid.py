#!/usr/bin/env python3
"""S19 arm A5: do independent fits of the same image differ as functions off the pixel grid?

docs/prereg/S19.md. Every INR is fitted to the 28x28 pixel grid only. Two fits of one image can
agree there and still be different functions elsewhere, which would be a nuisance at the level of
the realized function that no parameter symmetry accounts for. For pairs of fits of the same image
this evaluates both on the fit grid and on the interleaved off-grid points of the 55x55 grid (the
points whose row or column index is odd), and reports RMS discrepancies relative to the RMS of the
fitted image. The registered quantity is the median over images of the off-grid over on-grid ratio
for two independent fits (P-random-K, views k0 and k1). The other pairs are descriptive.

    .venv/bin/python scripts/81_s19_offgrid.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sirengap.data.schema import load_shard  # noqa: E402
from sirengap.eval.probes import evaluate_at  # noqa: E402
from sirengap.fitting.batched import make_coord_grid  # noqa: E402
from sirengap.models.params import SirenParams  # noqa: E402

N_IMAGES = 2000
SIDE, FINE = 28, 55
CORPORA = ROOT / "data" / "inrbench" / "mnist"


def load(corpus: Path, pattern: str = "shard_*.safetensors") -> tuple[SirenParams, pd.DataFrame]:
    shards = sorted(corpus.glob(pattern))
    ps = [load_shard(s) for s in shards]
    meta = pd.concat([pd.read_parquet(s.with_suffix(".parquet")) for s in shards], ignore_index=True)
    hidden = tuple((torch.cat([p.hidden[i][0] for p in ps]), torch.cat([p.hidden[i][1] for p in ps]))
                   for i in range(ps[0].n_layers))
    return SirenParams(hidden=hidden, w_out=torch.cat([p.w_out for p in ps]),
                       b_out=torch.cat([p.b_out for p in ps])), meta


def rows_for(meta: pd.DataFrame, ids: list[int]) -> torch.Tensor:
    pos = pd.Series(range(len(meta)), index=meta["image_id"].to_numpy())
    pos = pos[~pos.index.duplicated()]
    return torch.as_tensor(pos.loc[ids].to_numpy())


def take(p: SirenParams, idx: torch.Tensor) -> SirenParams:
    return SirenParams(hidden=tuple((w[idx], b[idx]) for w, b in p.hidden),
                       w_out=p.w_out[idx], b_out=p.b_out[idx])


@torch.no_grad()
def discrepancy(a: SirenParams, b: SirenParams, grid: torch.Tensor, off: torch.Tensor,
                chunk: int = 250) -> dict:
    on_r, off_r, scale = [], [], []
    for i in range(0, a.batch, chunk):
        idx = torch.arange(i, min(i + chunk, a.batch))
        pa, pb = take(a, idx), take(b, idx)
        fa_g, fb_g = evaluate_at(pa, grid), evaluate_at(pb, grid)
        fa_o, fb_o = evaluate_at(pa, off), evaluate_at(pb, off)
        on_r.append((fa_g - fb_g).pow(2).mean(dim=(1, 2)).sqrt())
        off_r.append((fa_o - fb_o).pow(2).mean(dim=(1, 2)).sqrt())
        scale.append(fa_g.pow(2).mean(dim=(1, 2)).sqrt())
    on, offd, s = torch.cat(on_r), torch.cat(off_r), torch.cat(scale)
    ratio = offd / on.clamp_min(1e-12)
    med = lambda t: float(t.median())  # noqa: E731
    return {"n": int(on.numel()), "on_grid_rel_median": med(on / s), "off_grid_rel_median": med(offd / s),
            "ratio_off_over_on_median": med(ratio),
            "ratio_q10_q90": [float(x) for x in torch.quantile(ratio, torch.tensor([0.1, 0.9]))]}


def main() -> None:
    grid = make_coord_grid(SIDE, SIDE)
    fine = make_coord_grid(FINE, FINE)
    i, j = torch.meshgrid(torch.arange(FINE), torch.arange(FINE), indexing="ij")
    off = fine[((i % 2 == 1) | (j % 2 == 1)).flatten()]

    k0, m0 = load(CORPORA / "P-random-K", "shard_k0_*.safetensors")
    k1, m1 = load(CORPORA / "P-random-K", "shard_k1_*.safetensors")
    ids = sorted(set(m0["image_id"]) & set(m1["image_id"]))[:N_IMAGES]
    out: dict = {"study": "S19 A5 off-grid functional discrepancy", "prereg": "docs/prereg/S19.md",
                 "n_images": len(ids), "fit_grid": SIDE, "fine_grid": FINE, "pairs": {}}
    out["pairs"]["independent_fits_randomK_k0_k1"] = discrepancy(
        take(k0, rows_for(m0, ids)), take(k1, rows_for(m1, ids)), grid, off)
    print("independent", out["pairs"]["independent_fits_randomK_k0_k1"], flush=True)

    shared, ms = load(CORPORA / "P-shared-det")
    random_, mr = load(CORPORA / "P-random")
    stoch, mt = load(CORPORA / "P-shared-stoch")
    common = sorted(set(ms["image_id"]) & set(mr["image_id"]) & set(mt["image_id"]))[:N_IMAGES]
    a = take(shared, rows_for(ms, common))
    out["pairs"]["shared_det_vs_random"] = discrepancy(a, take(random_, rows_for(mr, common)), grid, off)
    out["pairs"]["shared_det_vs_shared_stoch"] = discrepancy(a, take(stoch, rows_for(mt, common)), grid, off)
    shifted = common[1:] + common[:1]
    out["pairs"]["different_images_shared_det"] = discrepancy(a, take(shared, rows_for(ms, shifted)), grid, off)
    for k in ("shared_det_vs_random", "shared_det_vs_shared_stoch", "different_images_shared_det"):
        print(k, out["pairs"][k], flush=True)

    path = ROOT / "results" / "s19" / "offgrid_mnist.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

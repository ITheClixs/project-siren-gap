"""INR-Bench shard IO and metadata schema validation (Ch4, T8).

Shards: safetensors, one file per batch of INRs (canonical stored form).
Metadata: parquet, one row per INR, validated on every read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from sirengap.models.params import SirenParams

REQUIRED_COLUMNS: dict[str, str] = {
    "image_id": "int",
    "label": "int",
    "split": "str",
    "activation": "str",
    "protocol": "str",
    "init_seed": "int",
    "fit_seed": "int",
    "steps": "int",
    "lr": "float",
    "final_psnr": "float",
    "final_loss": "float",
    "wallclock_s": "float",
    "code_version": "str",
}
VALID_SPLITS = ("train", "val", "test")
SEED_DISJOINT_PROTOCOLS = ("P-random", "P-random-K")


class SchemaError(ValueError):
    """Raised when shard metadata violates the INR-Bench schema."""


def save_shard(params: SirenParams, path: str | Path) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for i, (w, b) in enumerate(params.hidden):
        tensors[f"hidden.{i}.W"] = w.contiguous().cpu()
        tensors[f"hidden.{i}.b"] = b.contiguous().cpu()
    tensors["w_out"] = params.w_out.contiguous().cpu()
    tensors["b_out"] = params.b_out.contiguous().cpu()
    save_file(tensors, str(path))


def load_shard(path: str | Path) -> SirenParams:
    tensors = load_file(str(path))
    n_layers = sum(1 for k in tensors if k.endswith(".W"))
    hidden = tuple((tensors[f"hidden.{i}.W"], tensors[f"hidden.{i}.b"]) for i in range(n_layers))
    return SirenParams(hidden=hidden, w_out=tensors["w_out"], b_out=tensors["b_out"])


def load_corpus(corpus_dir: str | Path) -> tuple[SirenParams, pd.DataFrame]:
    """Load every shard of a generated corpus into one batched SirenParams plus
    row-aligned metadata (validated). Shard order = sorted filename order."""
    corpus_dir = Path(corpus_dir)
    shards = sorted(corpus_dir.glob("shard_*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no shards in {corpus_dir}")
    params_list = [load_shard(p) for p in shards]
    metas = [pd.read_parquet(p.with_suffix(".parquet")) for p in shards]
    hidden = tuple(
        (
            torch.cat([p.hidden[i][0] for p in params_list]),
            torch.cat([p.hidden[i][1] for p in params_list]),
        )
        for i in range(params_list[0].n_layers)
    )
    params = SirenParams(
        hidden=hidden,
        w_out=torch.cat([p.w_out for p in params_list]),
        b_out=torch.cat([p.b_out for p in params_list]),
    )
    meta = pd.concat(metas, ignore_index=True)
    if len(meta) != params.batch:
        raise SchemaError(f"metadata rows {len(meta)} != params batch {params.batch}")
    validate_metadata(meta)
    return params, meta


def validate_metadata(df: pd.DataFrame) -> None:
    """Raise SchemaError listing every violation (T8). Checks: required columns,
    no NaNs, valid split values, image-id disjointness across splits, and seed
    disjointness across splits for random-init protocols."""
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    nan_cols = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
    if nan_cols:
        problems.append(f"NaNs in required columns: {nan_cols}")

    bad_splits = set(df["split"].unique()) - set(VALID_SPLITS)
    if bad_splits:
        problems.append(f"invalid split values: {sorted(bad_splits)}")

    for a in VALID_SPLITS:
        for b in VALID_SPLITS:
            if a >= b:
                continue
            ids_a = set(df.loc[df["split"] == a, "image_id"])
            ids_b = set(df.loc[df["split"] == b, "image_id"])
            overlap = ids_a & ids_b
            if overlap:
                problems.append(
                    f"image_id overlap between {a}/{b}: {len(overlap)} ids (e.g. {sorted(overlap)[:3]})"
                )

    rnd = df[df["protocol"].isin(SEED_DISJOINT_PROTOCOLS)]
    if not rnd.empty:
        test_seeds = set(
            map(tuple, rnd.loc[rnd["split"] == "test", ["init_seed", "fit_seed"]].values)
        )
        train_seeds = set(
            map(tuple, rnd.loc[rnd["split"] != "test", ["init_seed", "fit_seed"]].values)
        )
        overlap = test_seeds & train_seeds
        if overlap:
            problems.append(f"(init_seed, fit_seed) overlap train/test: {len(overlap)} pairs")

    if problems:
        raise SchemaError("; ".join(problems))

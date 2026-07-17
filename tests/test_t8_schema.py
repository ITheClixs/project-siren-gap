"""T8: metadata schema validation + shard IO round-trip + split disjointness."""

import pandas as pd
import pytest
import torch

from conftest import random_params
from sirengap.data.schema import SchemaError, load_shard, save_shard, validate_metadata


def _valid_df() -> pd.DataFrame:
    rows = []
    for i in range(12):
        split = "train" if i < 8 else ("val" if i < 10 else "test")
        rows.append(
            {
                "image_id": i,
                "label": i % 10,
                "split": split,
                "activation": "sine",
                "protocol": "P-random",
                "init_seed": 1000 + i,
                "fit_seed": 2000 + i,
                "steps": 1000,
                "lr": 1e-3,
                "final_psnr": 32.5,
                "final_loss": 1e-3,
                "wallclock_s": 0.5,
                "code_version": "abc123",
            }
        )
    return pd.DataFrame(rows)


def test_valid_metadata_passes() -> None:
    validate_metadata(_valid_df())


def test_missing_column_rejected() -> None:
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_metadata(_valid_df().drop(columns=["final_psnr"]))


def test_split_image_overlap_rejected() -> None:
    df = _valid_df()
    df.loc[11, "image_id"] = 0  # test row reuses a train image id
    with pytest.raises(SchemaError, match="image_id overlap"):
        validate_metadata(df)


def test_seed_overlap_rejected_for_random_protocols() -> None:
    df = _valid_df()
    df.loc[11, ["init_seed", "fit_seed"]] = df.loc[0, ["init_seed", "fit_seed"]].values
    df.loc[11, "image_id"] = 999
    with pytest.raises(SchemaError, match="overlap train/test"):
        validate_metadata(df)


def test_bad_split_value_rejected() -> None:
    df = _valid_df()
    df.loc[5, "split"] = "holdout"
    with pytest.raises(SchemaError, match="invalid split"):
        validate_metadata(df)


def test_nan_rejected() -> None:
    df = _valid_df()
    df.loc[3, "final_psnr"] = float("nan")
    with pytest.raises(SchemaError, match="NaNs"):
        validate_metadata(df)


def test_shard_roundtrip(tmp_path) -> None:
    params = random_params(5, 2, (16, 12), 3, seed=41)
    path = tmp_path / "shard.safetensors"
    save_shard(params, path)
    loaded = load_shard(path)
    assert torch.equal(params.flat(), loaded.flat())
    assert loaded.widths() == (16, 12)

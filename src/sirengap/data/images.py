"""Image dataset loaders (MNIST / FashionMNIST IDX, CIFAR-10 pickle) — stdlib + numpy only.

Raw files cached under data/ (gitignored). Targets returned in [-1, 1], flattened
row-major to match sirengap.fitting.batched.make_coord_grid.
"""

from __future__ import annotations

import gzip
import hashlib
import pickle
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

MIRRORS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "fashionmnist": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
}
IDX_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# verified against the archive fetched 2026-07-27; see docs/REPLICATION.md
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


def _md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324 — integrity, not a security boundary


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()  # noqa: S324 — integrity of a public archive, not a security boundary
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    """Download atomically: a partial transfer must never look like a finished file.

    The naive version (skip when `dest` exists) silently hands a truncated archive to the
    caller if a download dies or is read while still in flight — which produced a partial
    CIFAR-10 extraction (missing data_batch_1, truncated data_batch_2) on 2026-07-27.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_md5 is None or _md5(dest) == expected_md5:
            return dest
        print(f"{dest.name}: checksum mismatch, re-downloading")
        dest.unlink()
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, part)  # noqa: S310 — fixed https/s3 mirrors
        if expected_md5 is not None and _md5(part) != expected_md5:
            raise RuntimeError(f"checksum mismatch for {url}; refusing to use the download")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.rename(dest)
    return dest


def _parse_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    arr = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    return arr.reshape(dims)


def load_idx_dataset(name: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """name in {mnist, fashionmnist}, split in {train, test}.

    Returns (images [N, 784, 1] in [-1,1], labels [N] int64).
    """
    if name not in MIRRORS:
        raise ValueError(f"unknown dataset {name}")
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split}")
    base = MIRRORS[name]
    cache = DATA_DIR / name
    imgs = _parse_idx(_download(base + IDX_FILES[f"{split}_images"], cache / IDX_FILES[f"{split}_images"]))
    labels = _parse_idx(_download(base + IDX_FILES[f"{split}_labels"], cache / IDX_FILES[f"{split}_labels"]))
    x = torch.from_numpy(imgs.astype(np.float32) / 255.0 * 2.0 - 1.0).reshape(len(imgs), -1, 1)
    y = torch.from_numpy(labels.astype(np.int64))
    return x, y


CIFAR_MEMBERS = tuple([f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"])


def load_cifar10(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (images [N, 1024, 3] in [-1,1] row-major spatial, labels [N])."""
    tar_path = _download(CIFAR_URL, DATA_DIR / "cifar10" / "cifar-10-python.tar.gz", CIFAR_MD5)
    extract_dir = DATA_DIR / "cifar10"
    batch_dir = extract_dir / "cifar-10-batches-py"
    if not all((batch_dir / m).exists() for m in CIFAR_MEMBERS):
        # extract beside the target, then move into place, so an interrupted extraction
        # cannot leave a directory that later looks complete
        staging = extract_dir / "_staging"
        if staging.exists():
            shutil.rmtree(staging)
        with tarfile.open(tar_path) as tf:
            tf.extractall(staging, filter="data")
        missing = [m for m in CIFAR_MEMBERS if not (staging / "cifar-10-batches-py" / m).exists()]
        if missing:
            raise RuntimeError(f"CIFAR-10 archive is missing {missing}")
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        (staging / "cifar-10-batches-py").rename(batch_dir)
        shutil.rmtree(staging)
    names = [f"data_batch_{i}" for i in range(1, 6)] if split == "train" else ["test_batch"]
    xs, ys = [], []
    for n in names:
        with open(batch_dir / n, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(np.asarray(d[b"data"], dtype=np.uint8))
        ys.append(np.asarray(d[b"labels"], dtype=np.int64))
    raw = np.concatenate(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC
    x = torch.from_numpy(raw.astype(np.float32) / 255.0 * 2.0 - 1.0).reshape(len(raw), -1, 3)
    y = torch.from_numpy(np.concatenate(ys))
    return x, y


@dataclass(frozen=True)
class DatasetSpec:
    """Geometry and split layout of a source image dataset (docs/THINKING/G3-design.md).

    val_start is the index in the *train file* where the INR validation split begins;
    train = [0, val_start), val = [val_start, n_train_file). The official test file is
    addressed as TEST_ID_OFFSET + index by the generation script.
    """

    name: str
    side: int
    channels: int
    n_train_file: int
    val_start: int


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec("mnist", side=28, channels=1, n_train_file=60000, val_start=55000),
    "fashionmnist": DatasetSpec(
        "fashionmnist", side=28, channels=1, n_train_file=60000, val_start=55000
    ),
    # CIFAR-10 keeps the same 5000-image INR validation split, taken off a 50k train file.
    "cifar10": DatasetSpec("cifar10", side=32, channels=3, n_train_file=50000, val_start=45000),
    # Luminance CIFAR-10: identical images, identical geometry, c = 1 instead of c = 3. Exists to
    # separate "natural image statistics" from "output-channel count", which are confounded in
    # every comparison between the CIFAR-10 corpora and the grayscale ones (S1-gray prereg).
    "cifar10gray": DatasetSpec(
        "cifar10gray", side=32, channels=1, n_train_file=50000, val_start=45000
    ),
}

# ITU-R BT.601 luma, the standard RGB -> grayscale conversion
_LUMA = (0.299, 0.587, 0.114)


def spec_of(name: str) -> DatasetSpec:
    if name not in DATASET_SPECS:
        raise ValueError(f"unknown dataset {name}; known: {sorted(DATASET_SPECS)}")
    return DATASET_SPECS[name]


def load_dataset(name: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform loader: name in DATASET_SPECS, split in {train, test}.

    Returns (images [N, side*side, channels] in [-1,1] row-major, labels [N] int64).
    """
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split}")
    if name == "cifar10":
        return load_cifar10(split)
    if name == "cifar10gray":
        x, y = load_cifar10(split)
        # x is [N, P, 3] in [-1, 1]; luma is affine in the channels, so the range is preserved
        w = torch.tensor(_LUMA, dtype=x.dtype)
        return (x * w).sum(dim=2, keepdim=True), y
    return load_idx_dataset(name, split)

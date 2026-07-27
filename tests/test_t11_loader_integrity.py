"""T11: dataset downloads/extractions fail loudly instead of yielding partial data.

Regression test for 2026-07-27: the loader skipped the download whenever the destination
file existed, so a transfer that was still in flight (or had died) was treated as complete;
tarfile then extracted whatever prefix was readable and the corpus generator started fitting
INRs against a dataset that was missing data_batch_1 and had data_batch_2 truncated.
"""

from __future__ import annotations

import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

from sirengap.data import images


def _write_tar(path: Path, members: list[str], n_rows: int = 4) -> None:
    staged = path.parent / "cifar-10-batches-py"
    staged.mkdir(parents=True, exist_ok=True)
    for name in members:
        payload = {
            b"data": np.zeros((n_rows, 3072), dtype=np.uint8),
            b"labels": [0] * n_rows,
        }
        with open(staged / name, "wb") as f:
            pickle.dump(payload, f)
    with tarfile.open(path, "w:gz") as tf:
        tf.add(staged, arcname="cifar-10-batches-py")


def test_download_rejects_file_with_wrong_checksum(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "archive.bin"
    dest.write_bytes(b"truncated")
    good = b"the whole thing"

    def fake_urlretrieve(url: str, out: str) -> None:
        Path(out).write_bytes(good)

    monkeypatch.setattr(images.urllib.request, "urlretrieve", fake_urlretrieve)
    images._download("https://example.invalid/a", dest, images._md5_of_bytes(good))
    assert dest.read_bytes() == good


def test_download_leaves_no_file_when_the_transfer_dies(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "archive.bin"

    def dying_urlretrieve(url: str, out: str) -> None:
        Path(out).write_bytes(b"half")
        raise ConnectionError("link dropped")

    monkeypatch.setattr(images.urllib.request, "urlretrieve", dying_urlretrieve)
    with pytest.raises(ConnectionError):
        images._download("https://example.invalid/a", dest)
    assert not dest.exists(), "a dead transfer must not leave a file that looks complete"


def test_cifar_extraction_refuses_incomplete_archive(tmp_path, monkeypatch) -> None:
    tar_path = tmp_path / "cifar10" / "cifar-10-python.tar.gz"
    tar_path.parent.mkdir(parents=True)
    _write_tar(tar_path, ["data_batch_1", "test_batch", "batches.meta"])  # 2..5 missing

    monkeypatch.setattr(images, "DATA_DIR", tmp_path)
    monkeypatch.setattr(images, "_download", lambda url, dest, md5=None: tar_path)
    with pytest.raises(RuntimeError, match="missing"):
        images.load_cifar10("train")


def test_cifar_extraction_accepts_complete_archive(tmp_path, monkeypatch) -> None:
    tar_path = tmp_path / "cifar10" / "cifar-10-python.tar.gz"
    tar_path.parent.mkdir(parents=True)
    _write_tar(tar_path, list(images.CIFAR_MEMBERS))

    monkeypatch.setattr(images, "DATA_DIR", tmp_path)
    monkeypatch.setattr(images, "_download", lambda url, dest, md5=None: tar_path)
    x, y = images.load_cifar10("train")
    assert x.shape == (20, 1024, 3) and y.shape == (20,)
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0

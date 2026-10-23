import hashlib
from pathlib import Path

import pytest

from bibcut.weights import Weights, ensure


def _published(tmp_path: Path, payload: bytes) -> Weights:
    src = tmp_path / "published.onnx"
    src.write_bytes(payload)
    return Weights(
        name="toy",
        url=src.as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(),
        license="Apache-2.0",
        source="test",
    )


def test_downloads_and_verifies(tmp_path: Path) -> None:
    w = _published(tmp_path, b"model bytes")
    path = ensure(w, tmp_path / "cache")
    assert path.read_bytes() == b"model bytes"
    assert path.name == "toy.onnx"


def test_reuses_a_verified_file(tmp_path: Path) -> None:
    w = _published(tmp_path, b"model bytes")
    path = ensure(w, tmp_path / "cache")
    (tmp_path / "published.onnx").unlink()
    assert ensure(w, tmp_path / "cache") == path


def test_replaces_a_corrupted_file(tmp_path: Path) -> None:
    w = _published(tmp_path, b"model bytes")
    path = ensure(w, tmp_path / "cache")
    path.write_bytes(b"truncated")
    assert ensure(w, tmp_path / "cache").read_bytes() == b"model bytes"


def test_rejects_a_mismatching_download(tmp_path: Path) -> None:
    w = _published(tmp_path, b"model bytes")
    (tmp_path / "published.onnx").write_bytes(b"something else")
    with pytest.raises(ValueError, match="SHA-256"):
        ensure(w, tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []

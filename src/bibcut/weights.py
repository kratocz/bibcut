"""Model weights: pinned downloads, never committed or redistributed (SPEC §9)."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Weights:
    name: str
    url: str
    sha256: str
    license: str
    # Where the file comes from and how it relates to the official release.
    source: str


RFDETR_S = Weights(
    name="rfdetr_s_v142_512x512",
    url="https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_s_v142_512x512.onnx",
    sha256="fcd3c3c4e0d3d5533ee9c3a174deb7f02dfe4b042b6fdd3d29040f3c6da7845e",
    license="Apache-2.0",
    source=(
        "Third-party ONNX export of Roboflow RF-DETR Small (official weights are PyTorch only); "
        "see SPEC §11 for provenance and the fallback."
    ),
)


def models_dir() -> Path:
    """`$BIBCUT_MODELS_DIR`, else the user cache — independent of the working directory."""
    env = os.environ.get("BIBCUT_MODELS_DIR")
    return Path(env) if env else Path.home() / ".cache" / "bibcut" / "models"


def ensure(weights: Weights, directory: Path | None = None) -> Path:
    """Return a verified local copy of `weights`, downloading it if missing or corrupted."""
    directory = directory or models_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{weights.name}.onnx"
    if path.is_file() and _sha256(path) == weights.sha256:
        return path

    part = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(weights.url) as response, part.open("wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
        actual = _sha256(part)
        if actual != weights.sha256:
            raise ValueError(
                f"{weights.url}: SHA-256 {actual} does not match the pinned {weights.sha256}"
            )
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

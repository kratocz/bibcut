"""Swappable backends (AGENTS.md: detection/tracking, pose and number recognition).

Each backend lives behind one of these protocols so that a licence-incompatible or heavy
implementation can be replaced without touching the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

import numpy as np
import numpy.typing as npt

from bibcut.models import Detection, Frame


class PersonTracker(Protocol):
    """Detects and tracks people; yields (track id, detection) per frame.

    Backends that also estimate pose fill `Detection.keypoints` in the same pass.
    """

    def track(self, frames: Iterable[Frame]) -> Iterator[tuple[int, Detection]]: ...


class NumberRecognizer(Protocol):
    """Reads a start number from a torso crop.

    Returns candidate (text, confidence) pairs, best first; empty when nothing is legible.
    """

    def read(self, crop: npt.NDArray[np.uint8]) -> list[tuple[str, float]]: ...

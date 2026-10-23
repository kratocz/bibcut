"""Swappable backends (AGENTS.md: detection, pose and number recognition).

Each backend lives behind one of these protocols so that a licence-incompatible or heavy
implementation can be replaced without touching the pipeline. Tracking is bibcut's own
(`bibcut.track`) and works on any detector's output.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from bibcut.models import Detection, Frame


class PersonDetector(Protocol):
    """Detects people in one frame.

    `input_size` is the (width, height) the frame must be decoded to. Backends that also
    estimate pose fill `Detection.keypoints` in the same pass.
    """

    input_size: tuple[int, int]

    def detect(self, frame: Frame) -> list[Detection]: ...


class NumberRecognizer(Protocol):
    """Reads a start number from a torso crop.

    Returns candidate (text, confidence) pairs, best first; empty when nothing is legible.
    """

    def read(self, crop: npt.NDArray[np.uint8]) -> list[tuple[str, float]]: ...

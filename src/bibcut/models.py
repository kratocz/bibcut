"""Data passed between pipeline stages (SPEC §6).

All times are seconds on the video's presentation timeline (PTS), never frame index x fps.
Boxes and keypoints are in pixels of the decoded frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

BBox = tuple[float, float, float, float]
"""Axis-aligned box as (x1, y1, x2, y2)."""

Keypoint = tuple[float, float, float]
"""(x, y, confidence)."""


@dataclass(frozen=True)
class Frame:
    """One decoded frame."""

    # Position in the decoded (possibly sampled) sequence; use `pts` for time.
    index: int
    pts: float
    image: npt.NDArray[np.uint8] = field(repr=False)


@dataclass(frozen=True)
class Detection:
    """One person in one frame, optionally with pose keypoints from the same pass."""

    pts: float
    bbox: BBox
    score: float
    keypoints: tuple[Keypoint, ...] | None = None


@dataclass
class Tracklet:
    """One continuous appearance of one person, as produced by the tracker."""

    id: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.detections[0].pts

    @property
    def end(self) -> float:
        return self.detections[-1].pts


@dataclass(frozen=True)
class NumberReading:
    """A start number read from one torso crop."""

    pts: float
    text: str
    confidence: float


@dataclass(frozen=True)
class Appearance:
    """An interval credited to a rider in the output."""

    start: float
    end: float
    confidence: float
    source: Literal["auto", "manual"] = "auto"


@dataclass
class RiderTimeline:
    number: str
    name: str | None = None
    appearances: list[Appearance] = field(default_factory=list)

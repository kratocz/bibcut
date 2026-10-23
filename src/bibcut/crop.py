"""Torso crops for number reading (SPEC §6, step 4).

M1 has no pose model, so the torso is a horizontal band of the person box. The box spans head to
board or wheel, and the start number sits on the chest or back below the head.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bibcut.models import BBox


def torso_box(bbox: BBox, *, top: float = 0.15, bottom: float = 0.55, widen: float = 0.1) -> BBox:
    """Band from `top` to `bottom` of the box height, `widen` x width wider on each side."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (
        max(0.0, x1 - widen * w),
        y1 + top * h,
        min(1.0, x2 + widen * w),
        y1 + bottom * h,
    )


def crop(image: npt.NDArray[np.uint8], box: BBox) -> npt.NDArray[np.uint8]:
    """Pixels of a normalized box; at least 1x1 so callers need not special-case it."""
    h, w = image.shape[:2]
    x1 = min(int(box[0] * w), w - 1)
    y1 = min(int(box[1] * h), h - 1)
    x2 = max(round(box[2] * w), x1 + 1)
    y2 = max(round(box[3] * h), y1 + 1)
    return image[y1:y2, x1:x2]

"""Person detection with RF-DETR through onnxruntime (SPEC §6, step 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from bibcut.models import BBox, Detection, Frame

PERSON = 1  # COCO category id; RF-DETR's 91-way head is indexed by COCO id
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class RFDETR:
    """RF-DETR exported to ONNX with a fixed square input.

    The model expects the whole frame stretched to `input_size`; decode straight to that size
    (ffmpeg does the scaling), and the normalized boxes map back onto any resolution.
    """

    def __init__(self, path: Path, *, threshold: float = 0.1, nms_iou: float = 0.5) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.use_deterministic_compute = True
        self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        (inp,) = self._session.get_inputs()
        self._input = inp.name
        self.input_size: tuple[int, int] = (int(inp.shape[3]), int(inp.shape[2]))
        self.threshold = threshold
        self.nms_iou = nms_iou

    def detect(self, frame: Frame) -> list[Detection]:
        h, w = frame.image.shape[:2]
        if (w, h) != self.input_size:
            raise ValueError(f"frame is {w}x{h}, the model expects {self.input_size}")
        boxes, logits = self._session.run(None, {self._input: preprocess(frame.image)})
        return postprocess(boxes[0], logits[0], frame.pts, self.threshold, self.nms_iou)


def preprocess(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    x = (image.astype(np.float32) / 255 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def postprocess(
    boxes: npt.NDArray[np.float32],
    logits: npt.NDArray[np.float32],
    pts: float,
    threshold: float,
    nms_iou: float = 0.5,
) -> list[Detection]:
    """Turn per-query (cx, cy, w, h) boxes and class logits into person detections, best first."""
    scores = 1 / (1 + np.exp(-logits[:, PERSON].astype(np.float64)))
    keep = np.flatnonzero(scores >= threshold)
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    cx, cy, w, h = boxes[keep].astype(np.float64).T
    xyxy = np.clip(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1), 0, 1)

    out: list[Detection] = []
    kept: list[npt.NDArray[np.float64]] = []
    for box, score in zip(xyxy, scores[keep], strict=True):
        if any(iou(box, k) >= nms_iou for k in kept):
            continue
        kept.append(box)
        bbox: BBox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        out.append(Detection(pts=pts, bbox=bbox, score=float(score)))
    return out


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0

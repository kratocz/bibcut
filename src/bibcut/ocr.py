"""Start-number reading with the PP-OCRv5 English text recognizer (SPEC §6, steps 5-6).

The recognizer reads one line of text. A torso crop has the number somewhere in it, not
necessarily in the middle, so it is read in overlapping horizontal bands and the most confident
reading wins. Output is restricted to digits: the start number is all a crop is read for, and
masking the letters keeps sponsor logos and the event name from outscoring it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

HEIGHT = 48
"""Input height the recognizer was trained at; width follows the crop's aspect ratio."""

DIGIT_CLASSES = np.arange(1, 11)
"""Output classes of '0'..'9': the model's dictionary starts with the digits, and class 0 is
the CTC blank (`inference.yml` of the pinned export)."""


class PPOCR:
    """`NumberRecognizer` on onnxruntime (CPU, deterministic)."""

    def __init__(
        self,
        path: Path,
        *,
        band: float = 0.4,
        step: float = 0.1,
        max_digits: int = 3,
        threads: int | None = None,
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.use_deterministic_compute = True
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.band, self.step, self.max_digits = band, step, max_digits

    def read(self, crop: npt.NDArray[np.uint8]) -> list[tuple[str, float]]:
        best: dict[str, float] = {}
        for band in bands(crop, self.band, self.step):
            text, confidence = self.recognize(band)
            if 1 <= len(text) <= self.max_digits and confidence > best.get(text, 0.0):
                best[text] = confidence
        return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))

    def recognize(self, image: npt.NDArray[np.uint8]) -> tuple[str, float]:
        """Digits in one line of text and their mean confidence."""
        x = preprocess(image)
        probs = self.session.run(None, {self.input: x})[0][0]
        return ctc_digits(probs)


def bands(crop: npt.NDArray[np.uint8], band: float, step: float) -> list[npt.NDArray[np.uint8]]:
    """Horizontal strips `band` of the crop height tall, every `step`, skipping slivers."""
    h = crop.shape[0]
    tops = np.arange(0.0, 1.0 - band + 1e-9, step)
    strips = [crop[int(t * h) : int((t + band) * h)] for t in tops]
    return [s for s in strips if s.shape[0] >= 8]


def preprocess(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """RGB crop → normalized BGR NCHW tensor at the recognizer's input height."""
    h, w = image.shape[:2]
    width = max(16, round(HEIGHT * w / h))
    x = resize(image, HEIGHT, width)[:, :, ::-1]
    x = (x / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def ctc_digits(probs: npt.NDArray[np.float32]) -> tuple[str, float]:
    """Greedy CTC decoding over the blank and the digit classes only."""
    sub = probs[:, [0, *DIGIT_CLASSES]]
    best = sub.argmax(axis=1)
    text, confidences, previous = "", [], 0
    for t, k in enumerate(best):
        if k != 0 and k != previous:
            text += str(k - 1)
            confidences.append(float(sub[t, k]))
        previous = k
    return text, float(np.mean(confidences)) if confidences else 0.0


def resize(image: npt.NDArray[np.uint8], height: int, width: int) -> npt.NDArray[np.float32]:
    """Bilinear resize that widens its kernel when shrinking, so downscaling averages pixels
    instead of aliasing (the same filter as Pillow's BILINEAR)."""
    rows = _weights(image.shape[0], height)
    cols = _weights(image.shape[1], width)
    x = image.astype(np.float32)
    # Two separable passes; a single einsum over all four axes would loop naively.
    x = np.tensordot(rows, x, axes=(1, 0))
    return np.tensordot(x, cols, axes=(1, 1)).transpose(0, 2, 1)


def _weights(n_in: int, n_out: int) -> npt.NDArray[np.float32]:
    """(n_out, n_in) matrix of normalized triangle-filter weights."""
    scale = n_in / n_out
    support = max(scale, 1.0)
    centers = (np.arange(n_out) + 0.5) * scale
    taps = np.arange(n_in) + 0.5
    w = np.clip(1.0 - np.abs(taps[None, :] - centers[:, None]) / support, 0.0, None)
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)

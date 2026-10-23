import numpy as np
import pytest

from bibcut.ocr import HEIGHT, PPOCR, bands, ctc_digits, preprocess, resize


def _probs(classes: list[int], confidence: float = 0.9) -> np.ndarray:
    """One time step per class, 438-way like the real model."""
    p = np.full((len(classes), 438), (1 - confidence) / 437, np.float32)
    p[np.arange(len(classes)), classes] = confidence
    return p


def test_ctc_collapses_repeats_and_drops_blanks() -> None:
    # '2' '2' blank '4' '4' blank '5' → "245"
    assert ctc_digits(_probs([3, 3, 0, 5, 5, 0, 6])) == ("245", pytest.approx(0.9))


def test_ctc_keeps_a_repeated_digit_split_by_a_blank() -> None:
    assert ctc_digits(_probs([2, 0, 2]))[0] == "11"


def test_ctc_ignores_letters() -> None:
    # Class 11 is 'A': masked out, so the blank (or a digit) wins that step.
    p = _probs([11, 8])
    p[0, 0] = 0.05
    assert ctc_digits(p)[0] == "7"


def test_ctc_on_nothing_is_empty() -> None:
    assert ctc_digits(_probs([0, 0, 0])) == ("", 0.0)


def test_resize_keeps_a_flat_image_flat() -> None:
    image = np.full((97, 61, 3), 200, np.uint8)
    out = resize(image, 48, 30)
    assert out.shape == (48, 30, 3)
    assert out == pytest.approx(200.0)


def test_resize_to_the_same_size_is_identity() -> None:
    image = np.random.default_rng(0).integers(0, 256, (10, 7, 3), dtype=np.uint8)
    assert resize(image, 10, 7) == pytest.approx(image.astype(np.float32))


def test_downscale_averages_instead_of_sampling() -> None:
    # Alternating one-pixel stripes shrunk 4x: point sampling would hit one colour only.
    image = np.zeros((8, 8, 3), np.uint8)
    image[:, ::2] = 255
    out = resize(image, 2, 2)
    assert out.min() > 100 and out.max() < 155


def test_preprocess_shapes_and_normalizes() -> None:
    image = np.zeros((20, 40, 3), np.uint8)
    image[..., 0] = 255  # pure red
    x = preprocess(image)
    assert x.shape == (1, 3, HEIGHT, 96)
    assert x.dtype == np.float32
    # BGR order: red ends up in the last channel.
    assert x[0, 2] == pytest.approx(1.0)
    assert x[0, 0] == pytest.approx(-1.0)


def test_bands_cover_the_crop_and_skip_slivers() -> None:
    crop = np.zeros((100, 10, 3), np.uint8)
    strips = bands(crop, 0.4, 0.1)
    assert [s.shape[0] for s in strips] == [40] * 7
    assert bands(np.zeros((15, 10, 3), np.uint8), 0.4, 0.1) == []


class FakePPOCR(PPOCR):
    """Band logic without the model: reads a scripted text per band."""

    def __init__(self, script: list[tuple[str, float]]) -> None:
        self.band, self.step, self.max_digits = 0.4, 0.1, 3
        self.script = iter(script)

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        return next(self.script)


def test_read_keeps_the_best_confidence_per_text_best_first() -> None:
    crop = np.zeros((100, 10, 3), np.uint8)
    script = [("245", 0.5), ("", 0.0), ("2451", 0.99), ("45", 0.7), ("245", 0.8), ("", 0), ("", 0)]
    assert FakePPOCR(script).read(crop) == [("245", 0.8), ("45", 0.7)]

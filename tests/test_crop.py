import numpy as np
import pytest

from bibcut.crop import crop, torso_box


def test_torso_is_a_band_of_the_person_box() -> None:
    box = torso_box((0.2, 0.1, 0.3, 0.5), top=0.15, bottom=0.55, widen=0.0)
    assert box == pytest.approx((0.2, 0.16, 0.3, 0.32))


def test_torso_widening_is_clipped_to_the_frame() -> None:
    x1, _, x2, _ = torso_box((0.0, 0.1, 0.1, 0.5), widen=0.2)
    assert (x1, x2) == pytest.approx((0.0, 0.12))


def test_crop_maps_normalized_box_to_pixels() -> None:
    image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    out = crop(image, (0.25, 0.2, 0.5, 0.6))
    assert out.shape == (4, 5, 3)
    assert (out == image[2:6, 5:10]).all()


def test_crop_is_never_empty() -> None:
    image = np.zeros((10, 20, 3), np.uint8)
    assert crop(image, (0.5, 0.5, 0.5, 0.5)).shape == (1, 1, 3)

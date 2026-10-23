import numpy as np
import pytest

from bibcut.detect import PERSON, postprocess, preprocess


def _outputs(rows: list[tuple[tuple[float, float, float, float], int, float]]):
    """RF-DETR-shaped outputs: (cx, cy, w, h) boxes and 91-way logits, one query per row."""
    boxes = np.zeros((300, 4), np.float32)
    logits = np.full((300, 91), -10.0, np.float32)
    for i, (box, cls, prob) in enumerate(rows):
        boxes[i] = box
        logits[i, cls] = np.log(prob / (1 - prob))
    return boxes, logits


def test_keeps_confident_people_as_xyxy() -> None:
    boxes, logits = _outputs([((0.5, 0.5, 0.2, 0.4), PERSON, 0.9)])
    (det,) = postprocess(boxes, logits, pts=1.5, threshold=0.3)
    assert det.pts == 1.5
    assert det.bbox == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert det.score == pytest.approx(0.9)


def test_drops_other_classes_and_weak_scores() -> None:
    boxes, logits = _outputs([((0.5, 0.5, 0.2, 0.4), 3, 0.9), ((0.2, 0.2, 0.1, 0.1), PERSON, 0.2)])
    assert postprocess(boxes, logits, pts=0, threshold=0.3) == []


def test_suppresses_duplicates_keeping_the_stronger() -> None:
    boxes, logits = _outputs(
        [((0.5, 0.5, 0.2, 0.4), PERSON, 0.6), ((0.51, 0.5, 0.2, 0.4), PERSON, 0.8)]
    )
    (det,) = postprocess(boxes, logits, pts=0, threshold=0.3)
    assert det.score == pytest.approx(0.8)


def test_clips_to_the_frame() -> None:
    boxes, logits = _outputs([((0.02, 0.98, 0.1, 0.1), PERSON, 0.9)])
    (det,) = postprocess(boxes, logits, pts=0, threshold=0.3)
    assert det.bbox == pytest.approx((0.0, 0.93, 0.07, 1.0))


def test_output_is_sorted_by_score() -> None:
    boxes, logits = _outputs(
        [((0.2, 0.5, 0.1, 0.2), PERSON, 0.5), ((0.8, 0.5, 0.1, 0.2), PERSON, 0.95)]
    )
    assert [d.score for d in postprocess(boxes, logits, pts=0, threshold=0.3)] == pytest.approx(
        [0.95, 0.5]
    )


def test_preprocess_normalizes_to_nchw() -> None:
    image = np.full((4, 6, 3), 255, np.uint8)
    x = preprocess(image)
    assert x.shape == (1, 3, 4, 6)
    assert x.dtype == np.float32
    assert x[0, 0, 0, 0] == pytest.approx((1 - 0.485) / 0.229)

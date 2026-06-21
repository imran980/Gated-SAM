import numpy as np

from gated_sam.objectives import coarse_agreement, perturbation_consistency, predicted_iou


def _pred(mock, box):
    mock.set_image(np.zeros((128, 128, 3), np.uint8))
    return mock.predict_best(np.asarray(box))


def test_predicted_iou_returns_score(mock, rng):
    p = _pred(mock, [36, 36, 92, 92])
    assert predicted_iou(mock, p, rng) == p.score


def test_coarse_agreement_in_range(mock, rng):
    p = _pred(mock, [36, 36, 92, 92])
    val = coarse_agreement(mock, p, rng)
    assert 0.0 <= val <= 1.0
    assert val > 0.5  # the chosen mask agrees with its own thresholded logits


def test_consistency_high_for_containing_box(mock, rng):
    p = _pred(mock, [30, 30, 98, 98])  # fully contains the disk with margin
    val = perturbation_consistency(mock, p, rng, K=6, jitter=6)
    assert val > 0.9  # jitter does not change a mask that is a fixed point


def test_consistency_lower_for_cutting_box(mock, rng):
    good = perturbation_consistency(mock, _pred(mock, [30, 30, 98, 98]), rng, K=6, jitter=6)
    bad = perturbation_consistency(mock, _pred(mock, [0, 0, 60, 60]), rng, K=6, jitter=8)
    assert good > bad  # a box that slices the object is jitter-unstable

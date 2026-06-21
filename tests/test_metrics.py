import numpy as np
import pytest

from gated_sam.metrics import dice, hd95, iou, mask_to_box, mean_pairwise_iou


def test_dice_iou_identical():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    assert dice(m, m) == pytest.approx(1.0, abs=1e-4)
    assert iou(m, m) == pytest.approx(1.0, abs=1e-4)


def test_dice_iou_disjoint():
    a = np.zeros((20, 20), bool); a[0:5, 0:5] = True
    b = np.zeros((20, 20), bool); b[10:15, 10:15] = True
    assert dice(a, b) < 1e-3
    assert iou(a, b) < 1e-3


def test_iou_half_overlap():
    a = np.zeros((10, 10), bool); a[:, :6] = True   # 60 px
    b = np.zeros((10, 10), bool); b[:, 4:] = True   # 60 px, overlap cols 4,5 => 20
    assert abs(iou(a, b) - 20 / 100) < 1e-6


def test_mean_pairwise_iou_identical_is_one():
    m = np.zeros((10, 10), bool); m[2:8, 2:8] = True
    assert mean_pairwise_iou([m, m, m]) == pytest.approx(1.0, abs=1e-4)


def test_hd95_zero_for_identical():
    m = np.zeros((30, 30), bool); m[5:25, 5:25] = True
    assert hd95(m, m) == 0.0


def test_hd95_nan_for_empty():
    a = np.zeros((10, 10), bool); a[2:5, 2:5] = True
    assert np.isnan(hd95(a, np.zeros((10, 10), bool)))


def test_mask_to_box_pad_and_clip():
    m = np.zeros((50, 50), bool); m[10:20, 15:30] = True
    box = mask_to_box(m, pad=5, shape=(50, 50))
    assert list(box) == [10, 5, 34, 24]
    assert mask_to_box(np.zeros((5, 5), bool)) is None

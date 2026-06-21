"""Segmentation metrics: Dice, IoU, HD95. Pure numpy/scipy — no GPU, fully testable."""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + _EPS))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return float(2 * inter / (total + _EPS))


def mean_pairwise_iou(masks: list[np.ndarray]) -> float:
    """Mean IoU over all unordered pairs — the core of perturbation-consistency."""
    n = len(masks)
    if n < 2:
        return 1.0  # a single mask is trivially self-consistent
    vals = [iou(masks[i], masks[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


def _boundary_points(mask: np.ndarray) -> np.ndarray | None:
    from scipy.ndimage import binary_erosion

    m = mask.astype(bool)
    if m.sum() == 0:
        return None
    edge = m ^ binary_erosion(m)
    pts = np.argwhere(edge)
    return pts if len(pts) else np.argwhere(m)


def hd95(a: np.ndarray, b: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance (pixels).

    Returns NaN if either mask is empty (caller aggregates with nan-aware stats and
    reports a separate failure count). This avoids silently rewarding empty masks.
    """
    from scipy.spatial import cKDTree

    pa = _boundary_points(a)
    pb = _boundary_points(b)
    if pa is None or pb is None:
        return float("nan")
    d_ab = cKDTree(pb).query(pa)[0]
    d_ba = cKDTree(pa).query(pb)[0]
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))


def mask_to_box(mask: np.ndarray, pad: int = 0, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """Tight [x1, y1, x2, y2] box around a mask, optionally padded and clipped."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    h, w = shape if shape is not None else mask.shape
    return np.array([
        max(0, xs.min() - pad),
        max(0, ys.min() - pad),
        min(w - 1, xs.max() + pad),
        min(h - 1, ys.max() + pad),
    ], dtype=int)

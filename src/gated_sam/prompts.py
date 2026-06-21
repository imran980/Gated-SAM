"""Box-prompt utilities: noise injection, jitter, and morphological box variants."""
from __future__ import annotations

import numpy as np

from .metrics import mask_to_box


def clip_box(box: np.ndarray, h: int, w: int) -> np.ndarray:
    """Clamp a box to the image and keep x2>x1, y2>y1."""
    b = box.astype(float).copy()
    b[0] = np.clip(b[0], 0, w - 2)
    b[1] = np.clip(b[1], 0, h - 2)
    b[2] = np.clip(b[2], b[0] + 1, w - 1)
    b[3] = np.clip(b[3], b[1] + 1, h - 1)
    return b.astype(int)


def add_box_noise(box: np.ndarray, noise: int, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Perturb a box by U(-noise, noise) on each coordinate (the paper's protocol)."""
    if noise == 0:
        return box.astype(int).copy()
    delta = rng.integers(-noise, noise + 1, size=4)
    return clip_box(box.astype(float) + delta, h, w)


def jitter_box(box: np.ndarray, jitter: int, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Small symmetric perturbation used by the consistency probe / candidate set."""
    return add_box_noise(box, jitter, h, w, rng)


def expand_box(box: np.ndarray, px: int, h: int, w: int) -> np.ndarray:
    return clip_box(box.astype(float) + np.array([-px, -px, px, px]), h, w)


def shrink_box(box: np.ndarray, px: int, h: int, w: int) -> np.ndarray:
    return clip_box(box.astype(float) + np.array([px, px, -px, -px]), h, w)


def candidate_boxes(
    mask: np.ndarray,
    h: int,
    w: int,
    *,
    pad: int,
    dilate_px,
    erode_px,
    n_jitter: int,
    jitter_px: int,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    """Build the labelled candidate neighborhood around the current mask's tight box.

    {tight box, dilate±, erode±, K jittered boxes}. Returns (source_label, box) so the
    trajectory can record *why* each candidate was proposed.
    """
    tight = mask_to_box(mask, pad=pad, shape=(h, w))
    if tight is None:
        return []
    out: list[tuple[str, np.ndarray]] = [("tight", tight)]
    for d in dilate_px:
        out.append((f"dilate+{d}", expand_box(tight, d, h, w)))
    for e in erode_px:
        out.append((f"erode-{e}", shrink_box(tight, e, h, w)))
    for k in range(n_jitter):
        out.append((f"jitter{k}", jitter_box(tight, jitter_px, h, w, rng)))
    return out

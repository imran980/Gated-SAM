"""Predictor abstraction over SAM / MedSAM, plus a GPU-free MockPredictor for tests.

Everything downstream (objectives, search, baselines) talks to this `Predictor`
interface only, so the exact same code runs against real SAM on an A100 and against
the deterministic mock in unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

LOWRES = 256  # SAM's low-resolution logit grid


@dataclass
class Prediction:
    """One chosen mask and everything needed to refine from it or score it."""
    mask: np.ndarray          # bool, (H, W)
    score: float              # SAM predicted-IoU for this mask
    logits: np.ndarray        # float, (LOWRES, LOWRES) — reusable as a dense mask prompt
    box: np.ndarray           # the [x1,y1,x2,y2] box that produced it
    source: str = "sam"       # provenance label for the trajectory


def _resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    im = Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)
    return np.array(im) > 127


class Predictor:
    """Interface. Concrete subclasses implement `_set_image` and `predict_all`."""

    def set_image(self, image: np.ndarray) -> None:
        # Re-embedding the image is the expensive part of SAM. Methods call set_image
        # defensively, so skip it when the same image array is already loaded; this keeps
        # one encoder pass per (predictor, image) and preserves the consistency-probe cache.
        if getattr(self, "_img_id", None) == id(image):
            return
        self._img_id = id(image)
        self._cache: dict = {}
        self._h, self._w = image.shape[:2]
        self._set_image(image)

    # subclasses override
    def _set_image(self, image: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_all(self, box: np.ndarray, mask_input: np.ndarray | None = None) -> list[Prediction]:  # pragma: no cover
        raise NotImplementedError

    # shared helpers ------------------------------------------------------
    def predict_best(self, box: np.ndarray, mask_input: np.ndarray | None = None) -> Prediction:
        preds = self.predict_all(box, mask_input)
        return max(preds, key=lambda p: p.score)

    def predict_best_cached(self, box: np.ndarray) -> Prediction:
        """Hint-free best prediction with an in-image cache (consistency probes repeat)."""
        key = tuple(int(v) for v in box)
        if key not in self._cache:
            self._cache[key] = self.predict_best(np.asarray(box))
        return self._cache[key]


class SamPredictorWrapper(Predictor):
    def __init__(self, sam_predictor):
        self._sam = sam_predictor

    def _set_image(self, image: np.ndarray) -> None:
        self._sam.set_image(image)

    def predict_all(self, box: np.ndarray, mask_input: np.ndarray | None = None) -> list[Prediction]:
        box = np.asarray(box).reshape(1, 4)
        kwargs = {"box": box, "multimask_output": True}
        if mask_input is not None:
            kwargs["mask_input"] = mask_input[None] if mask_input.ndim == 2 else mask_input
        masks, scores, lowres = self._sam.predict(**kwargs)
        return [
            Prediction(mask=masks[i].astype(bool), score=float(scores[i]),
                       logits=lowres[i].astype(np.float32), box=box.reshape(4))
            for i in range(len(scores))
        ]


def load_sam(model_type: str, checkpoint: str, device: str = "cuda") -> SamPredictorWrapper:
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictorWrapper(SamPredictor(sam))


def load_medsam(model_type: str, checkpoint: str, device: str = "cuda") -> SamPredictorWrapper:
    # MedSAM ships as a SAM ViT-B checkpoint; loaded the same way.
    return load_sam(model_type, checkpoint, device)


# ── Mock predictor (no torch, deterministic) ─────────────────────────────
class MockPredictor(Predictor):
    """Deterministic stand-in for SAM driven by a hidden ground-truth disk.

    `predict_all(box)` returns three candidates derived from the disk clipped to the
    box. Tighter, better-centered boxes yield masks that are stable under jitter
    (high perturbation-consistency) and closer to the disk (high true Dice), so the
    prompt-space search has a real, monotone signal to optimize — exactly the regime
    the method targets.
    """

    def __init__(self, h: int = 128, w: int = 128, center=(64, 64), radius: int = 28):
        self.h, self.w = h, w
        self.cx, self.cy = center
        self.r = radius
        yy, xx = np.mgrid[0:h, 0:w]
        self.disk = ((xx - self.cx) ** 2 + (yy - self.cy) ** 2) <= self.r ** 2

    def _set_image(self, image: np.ndarray) -> None:
        pass

    def _box_mask(self, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        m = np.zeros((self.h, self.w), dtype=bool)
        m[max(0, y1):y2 + 1, max(0, x1):x2 + 1] = True
        return m

    def _to_logits(self, mask: np.ndarray) -> np.ndarray:
        field = np.where(mask, 1.0, -1.0).astype(np.float32)
        return np.array(Image.fromarray(((field + 1) * 127).astype(np.uint8))
                        .resize((LOWRES, LOWRES), Image.NEAREST), dtype=np.float32) / 127.0 - 1.0

    def predict_all(self, box: np.ndarray, mask_input: np.ndarray | None = None) -> list[Prediction]:
        from scipy.ndimage import binary_erosion

        box = np.asarray(box, dtype=int)
        bm = self._box_mask(box)
        a = self.disk & bm                      # good: disk clipped to box
        c = binary_erosion(a, iterations=3) if a.sum() > 0 else a
        b = bm                                   # degenerate: the whole box
        cands = [
            ("disk_in_box", a, 0.90),
            ("eroded", c, 0.70),
            ("box_fill", b, 0.50),
        ]
        return [
            Prediction(mask=m.astype(bool), score=s, logits=self._to_logits(m),
                       box=box.astype(int), source=name)
            for name, m, s in cands
        ]

"""The methods compared in the main table.

The contribution is proven by one ordering:  Ours > predicted-IoU gate > ungated.
So every method below shares the same prompts, predictor, and seeds; they differ only
in how (or whether) they refine and select.
"""
from __future__ import annotations

import numpy as np

from .metrics import mask_to_box
from .models import Predictor
from .objectives import build_objective
from .prompts import clip_box
from .refine import refine_search


def _one_pass_T(predictor: Predictor, image: np.ndarray, init_box: np.ndarray, cfg):
    """A single application of the refinement map T: returns (pass1, pass2) predictions."""
    h, w = image.shape[:2]
    box = clip_box(np.asarray(init_box), h, w)
    p1 = predictor.predict_best(box)
    tight = mask_to_box(p1.mask, pad=int(cfg.search.box_pad), shape=(h, w))
    if tight is None:
        return p1, p1
    hint = p1.logits if bool(cfg.search.use_mask_hint) else None
    p2 = predictor.predict_best(tight, mask_input=hint)
    return p1, p2


def vanilla_sam(predictor, image, init_box, rng, cfg):
    predictor.set_image(image)
    h, w = image.shape[:2]
    return predictor.predict_best(clip_box(np.asarray(init_box), h, w)).mask


def medsam(predictor, image, init_box, rng, cfg):
    # Same call as vanilla SAM but with the MedSAM predictor passed in.
    return vanilla_sam(predictor, image, init_box, rng, cfg)


def ungated_cascade(predictor, image, init_box, rng, cfg):
    """One pass of T, no gate — always take the refined mask (the PerSAM-style baseline)."""
    predictor.set_image(image)
    _, p2 = _one_pass_T(predictor, image, init_box, cfg)
    return p2.mask


def predicted_iou_gate(predictor, image, init_box, rng, cfg):
    """The OLD method: refine once, keep whichever pass has higher SAM predicted-IoU."""
    predictor.set_image(image)
    p1, p2 = _one_pass_T(predictor, image, init_box, cfg)
    return (p2 if p2.score > p1.score else p1).mask


def ours(predictor, image, init_box, rng, cfg):
    """Consistency-driven prompt-space search with guarded return."""
    objective = build_objective(cfg)
    return refine_search(predictor, image, init_box, objective, cfg, rng).mask


# Methods that use the SAM predictor vs the MedSAM predictor.
SAM_METHODS = {
    "vanilla_sam": vanilla_sam,
    "ungated_cascade": ungated_cascade,
    "predicted_iou_gate": predicted_iou_gate,
    "ours": ours,
}
MEDSAM_METHODS = {"medsam": medsam}

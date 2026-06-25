"""The methods compared in the main table.

The contribution is proven by one ordering:  Ours > predicted-IoU gate > ungated.
So every method below shares the same prompts, predictor, and seeds; they differ only
in how (or whether) they refine and select.
"""
from __future__ import annotations

import numpy as np

from .metrics import iou, mask_to_box
from .models import Predictor
from .objectives import build_objective
from .prompts import clip_box
from .refine import refine_search


def _plausible(mask: np.ndarray, lo: float = 0.002, hi: float = 0.95) -> bool:
    """Reject degenerate masks (near-empty slivers or whole-image blobs)."""
    r = mask.sum() / mask.size
    return lo <= r <= hi


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


def consistency_gate(predictor, image, init_box, rng, cfg):
    """Ours: a guarded, anchored reference-free GATE between vanilla and one refinement.

    Refine once (the strong ungated move), then accept the refined mask ONLY IF the
    reference-free objective improves by a margin AND the refined mask stays on the same
    object (overlap with the vanilla mask) AND is non-degenerate. Otherwise keep vanilla.

    This restores the no-regression property the free search lost: the candidate set is
    just {vanilla, one anchored refinement}, both tied to the prompted object, so the
    objective cannot be gamed by a far-away stable blob. A clean prompt is already stable
    (margin not cleared -> keep vanilla); a noisy prompt's refinement is more stable
    (margin cleared -> refine).
    """
    predictor.set_image(image)
    h, w = image.shape[:2]
    box = clip_box(np.asarray(init_box), h, w)
    p0 = predictor.predict_best(box)
    tight = mask_to_box(p0.mask, pad=int(cfg.search.box_pad), shape=(h, w))
    if tight is None or not _plausible(p0.mask):
        return p0.mask
    hint = p0.logits if bool(cfg.search.use_mask_hint) else None
    p1 = predictor.predict_best(tight, mask_input=hint)

    obj = build_objective(cfg)
    q0 = obj(predictor, p0, rng)
    q1 = obj(predictor, p1, rng)
    margin = float(cfg.search.get("gate_margin", 0.03))
    anchor = float(cfg.search.get("gate_anchor", 0.30))
    accept = (q1 > q0 + margin) and (iou(p1.mask, p0.mask) >= anchor) and _plausible(p1.mask)
    return p1.mask if accept else p0.mask


def ours_search(predictor, image, init_box, rng, cfg):
    """Ablation: the free consistency-driven prompt-space search (over-optimizes Q)."""
    return refine_search(predictor, image, init_box, build_objective(cfg), cfg, rng).mask


# `ours` is the gate; the free search is kept for the ablation that explains its failure.
ours = consistency_gate

# Methods that use the SAM predictor vs the MedSAM predictor.
SAM_METHODS = {
    "vanilla_sam": vanilla_sam,
    "ungated_cascade": ungated_cascade,
    "predicted_iou_gate": predicted_iou_gate,
    "ours": consistency_gate,
}
MEDSAM_METHODS = {"medsam": medsam}

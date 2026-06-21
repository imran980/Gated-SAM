"""Prompt-space optimization: the refinement map T and the guarded consistency search.

T: prompt -> mask -> (tight box + logit hint) -> prompt.
The search builds a candidate neighborhood at each step, moves to the argmax of the
reference-free objective Q, and returns the best-Q mask seen across the WHOLE
trajectory (including the original single-pass prediction). That guarded return is
what gives the no-regression guarantee and fixes the delta=0 lock-in for free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import iou
from .models import Predictor, Prediction
from .objectives import Objective
from .prompts import candidate_boxes, clip_box


@dataclass
class Step:
    k: int
    source: str            # which candidate was chosen at this step
    Q: float               # objective value of the chosen mask
    score: float           # SAM predicted-IoU of the chosen mask
    move: float            # IoU(M_k, M_{k-1}) — step movement, for lock-in analysis
    n_candidates: int
    box: np.ndarray


@dataclass
class SearchResult:
    mask: np.ndarray                 # guarded best-Q mask (what you return / score)
    Q: float
    chosen_step: int                 # trajectory index that won the guard
    trajectory: list[Step] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)  # per-step masks (lock-in figures)

    @property
    def n_steps(self) -> int:
        return len(self.trajectory) - 1

    @property
    def moved(self) -> bool:
        """Did the guard leave the original single-pass prediction?"""
        return self.chosen_step != 0


def _candidates(predictor: Predictor, mask: np.ndarray, hint, h, w, scfg, rng) -> list[tuple[str, Prediction]]:
    """Predict every box in the neighborhood; include the 3 multimask outputs of the tight box."""
    boxes = candidate_boxes(
        mask, h, w,
        pad=int(scfg.box_pad), dilate_px=list(scfg.dilate_px), erode_px=list(scfg.erode_px),
        n_jitter=int(scfg.n_jitter_candidates), jitter_px=int(scfg.jitter_px), rng=rng,
    )
    out: list[tuple[str, Prediction]] = []
    for label, box in boxes:
        if label == "tight":
            for pr in predictor.predict_all(box, mask_input=hint):   # the 3 multimask outputs
                out.append((f"tight:{pr.source}", pr))
        else:
            out.append((label, predictor.predict_best(box, mask_input=hint)))
    return [(lab, pr) for lab, pr in out if pr.mask.sum() > 0]


def refine_search(predictor: Predictor, image: np.ndarray, init_box: np.ndarray,
                  objective: Objective, cfg, rng: np.random.Generator) -> SearchResult:
    """Consistency-driven prompt-space search with guarded return."""
    scfg = cfg.search
    predictor.set_image(image)
    h, w = image.shape[:2]
    box = clip_box(np.asarray(init_box), h, w)

    # Step 0: the original single-pass prediction (== vanilla SAM). Always in the guard.
    p = predictor.predict_best(box)
    Q = objective(predictor, p, rng)
    traj = [Step(0, "init", Q, p.score, 1.0, 1, p.box)]
    masks = [p.mask]
    best = (p, Q, 0)

    current, Q_cur = p, Q
    for k in range(1, int(scfg.max_steps) + 1):
        if current.mask.sum() == 0:
            break
        hint = current.logits if bool(scfg.use_mask_hint) else None
        cands = _candidates(predictor, current.mask, hint, h, w, scfg, rng)
        if not cands:
            break
        scored = [(lab, pr, objective(predictor, pr, rng)) for lab, pr in cands]
        lab, pr, Qc = max(scored, key=lambda t: t[2])
        move = iou(pr.mask, current.mask)
        traj.append(Step(k, lab, Qc, pr.score, move, len(cands), pr.box))
        masks.append(pr.mask)
        if Qc > best[1]:
            best = (pr, Qc, k)
        if Qc <= Q_cur + float(scfg.improve_eps):   # Q stopped improving -> stop
            current, Q_cur = pr, Qc
            break
        current, Q_cur = pr, Qc

    if not bool(scfg.guard):
        # Ablation: no guard -> return the last visited mask (can regress).
        last = traj[-1]
        return SearchResult(mask=masks[-1], Q=last.Q, chosen_step=last.k, trajectory=traj, masks=masks)

    bp, bQ, bk = best
    return SearchResult(mask=bp.mask, Q=bQ, chosen_step=bk, trajectory=traj, masks=masks)

"""Reference-free quality objectives Q(M). No ground truth, no auxiliary models.

These are the candidate signals compared in the Day-1 go/no-go and the objective the
prompt-space search maximizes. Each is callable as obj(predictor, prediction, rng) so
the search can treat them interchangeably (Day-6 objective ablation).

  predicted_iou           : SAM's own predicted-IoU head (the OLD gate / weak baseline).
  coarse_agreement        : IoU between thresholded coarse logits and the final mask.
  perturbation_consistency: mean pairwise IoU of masks from K jittered boxes (the method).
"""
from __future__ import annotations

import numpy as np

from .metrics import iou, mean_pairwise_iou
from .models import Predictor, Prediction
from .prompts import jitter_box


def predicted_iou(predictor: Predictor, pred: Prediction, rng: np.random.Generator) -> float:
    return float(pred.score)


def coarse_agreement(predictor: Predictor, pred: Prediction, rng: np.random.Generator) -> float:
    from .models import _resize_bool

    coarse = pred.logits > 0
    coarse = _resize_bool(coarse, pred.mask.shape)
    return iou(pred.mask, coarse)


def perturbation_consistency(
    predictor: Predictor, pred: Prediction, rng: np.random.Generator,
    K: int = 6, jitter: int = 8,
) -> float:
    """Run K jittered boxes around pred.box; return mean pairwise IoU of the masks.

    A mask that survives prompt jitter is a fixed point of the prompt->mask map and
    is empirically a better quality proxy than the predicted-IoU head under domain
    shift. Uses the predictor's per-image cache so repeated probes are free.
    """
    h, w = pred.mask.shape
    masks = []
    for _ in range(K):
        jb = jitter_box(pred.box, jitter, h, w, rng)
        masks.append(predictor.predict_best_cached(jb).mask)
    return mean_pairwise_iou(masks)


class Objective:
    """Bundles an objective with its hyper-parameters and a stable name."""

    def __init__(self, name: str, fn, **kw):
        self.name = name
        self._fn = fn
        self._kw = kw

    def __call__(self, predictor: Predictor, pred: Prediction, rng: np.random.Generator) -> float:
        return float(self._fn(predictor, pred, rng, **self._kw))


class ComboObjective(Objective):
    """Convex combination of normalized sub-objectives (Day-6 ablation: objective='combo')."""

    def __init__(self, parts: dict[str, Objective], weights: dict[str, float]):
        self.name = "combo"
        self.parts = parts
        total = sum(weights.get(k, 0.0) for k in parts) or 1.0
        self.weights = {k: weights.get(k, 0.0) / total for k in parts}

    def __call__(self, predictor: Predictor, pred: Prediction, rng: np.random.Generator) -> float:
        return float(sum(self.weights[k] * obj(predictor, pred, rng) for k, obj in self.parts.items()))


def build_objective(cfg) -> Objective:
    """Construct the Objective named in cfg.objective.name."""
    K = int(cfg.objective.get("consistency_K", 6))
    j = int(cfg.objective.get("consistency_jitter", 8))
    registry = {
        "predicted_iou": Objective("predicted_iou", predicted_iou),
        "coarse_agreement": Objective("coarse_agreement", coarse_agreement),
        "perturbation_consistency": Objective(
            "perturbation_consistency", perturbation_consistency, K=K, jitter=j),
    }
    name = cfg.objective.name
    if name == "combo":
        weights = dict(cfg.objective.get("combo_weights", {}))
        return ComboObjective(registry, weights)
    if name not in registry:
        raise ValueError(f"unknown objective {name!r}; choose from {list(registry) + ['combo']}")
    return registry[name]


def all_signal_objectives(cfg) -> dict[str, Objective]:
    """The three signals to correlate against true Dice in Day-1 (Figure 2)."""
    K = int(cfg.objective.get("consistency_K", 6))
    j = int(cfg.objective.get("consistency_jitter", 8))
    return {
        "predicted_iou": Objective("predicted_iou", predicted_iou),
        "coarse_agreement": Objective("coarse_agreement", coarse_agreement),
        "perturbation_consistency": Objective(
            "perturbation_consistency", perturbation_consistency, K=K, jitter=j),
    }

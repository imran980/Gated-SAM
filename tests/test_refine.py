import numpy as np

from gated_sam.metrics import dice
from gated_sam.objectives import build_objective
from gated_sam.refine import refine_search


def _init_dice(mock, disk_image, box):
    mock.set_image(disk_image)
    return dice(mock.predict_best(np.asarray(box)).mask, mock.disk)


def test_guard_never_regresses_Q(mock, disk_image, cfg, rng):
    """Core guarantee: returned Q >= Q of the original single-pass prediction."""
    obj = build_objective(cfg)
    for box in ([28, 28, 100, 100], [20, 20, 78, 78], [0, 0, 64, 64]):
        res = refine_search(mock, disk_image, np.array(box), obj, cfg, rng)
        assert res.Q >= res.trajectory[0].Q - 1e-9


def test_clean_prompt_stays_put(mock, disk_image, cfg, rng):
    """delta=0 case: a clean, jitter-stable prompt is already optimal -> guard keeps step 0."""
    obj = build_objective(cfg)
    res = refine_search(mock, disk_image, np.array([28, 28, 100, 100]), obj, cfg, rng)
    assert res.chosen_step == 0
    assert dice(res.mask, mock.disk) > 0.95


def test_recovery_from_bad_box(mock, disk_image, cfg, rng):
    """A box that slices the object is corrected by the search (recovery)."""
    box = np.array([20, 20, 78, 78])
    init = _init_dice(mock, disk_image, box)
    res = refine_search(mock, disk_image, box, build_objective(cfg), rng=rng, cfg=cfg)
    final = dice(res.mask, mock.disk)
    assert final >= init - 1e-6          # never worse than the starting mask
    assert final > init                   # and strictly recovers on this case


def test_trajectory_is_logged(mock, disk_image, cfg, rng):
    res = refine_search(mock, disk_image, np.array([20, 20, 78, 78]), build_objective(cfg), cfg, rng)
    assert len(res.masks) == len(res.trajectory)
    s = res.trajectory[0]
    assert s.k == 0 and s.source == "init"
    for step in res.trajectory[1:]:
        assert 0.0 <= step.move <= 1.0
        assert step.n_candidates >= 1


def test_guard_off_returns_last_step(mock, disk_image, cfg, rng):
    cfg.search["guard"] = False
    res = refine_search(mock, disk_image, np.array([20, 20, 78, 78]), build_objective(cfg), cfg, rng)
    assert res.mask.sum() > 0
    assert res.chosen_step == res.trajectory[-1].k

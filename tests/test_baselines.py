import numpy as np

from gated_sam.baselines import SAM_METHODS
from gated_sam.metrics import dice


def test_all_methods_return_nonempty(mock, disk_image, cfg, rng):
    box = np.array([24, 24, 104, 104])
    for name, fn in SAM_METHODS.items():
        mask = fn(mock, disk_image, box, rng, cfg)
        assert mask.sum() > 0, name
        assert dice(mask, mock.disk) > 0.5, name


def test_ours_not_worse_than_vanilla_on_bad_box(mock, disk_image, cfg, rng):
    box = np.array([18, 18, 76, 76])
    vanilla = SAM_METHODS["vanilla_sam"](mock, disk_image, box, rng, cfg)
    ours = SAM_METHODS["ours"](mock, disk_image, box, rng, cfg)
    assert dice(ours, mock.disk) >= dice(vanilla, mock.disk) - 1e-6

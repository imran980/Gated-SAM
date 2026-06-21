import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gated_sam.config import Config  # noqa: E402
from gated_sam.models import MockPredictor  # noqa: E402


@pytest.fixture
def cfg():
    return Config({
        "search": {
            "max_steps": 3, "box_pad": 6, "dilate_px": [8, 16], "erode_px": [6, 12],
            "n_jitter_candidates": 4, "jitter_px": 6, "use_mask_hint": False,
            "guard": True, "improve_eps": 0.005,
        },
        "objective": {
            "name": "perturbation_consistency", "consistency_K": 6,
            "consistency_jitter": 6, "combo_weights": {},
        },
    })


@pytest.fixture
def mock():
    return MockPredictor(h=128, w=128, center=(64, 64), radius=28)


@pytest.fixture
def disk_image():
    # The mock ignores pixels, but the API takes an HxWx3 image.
    return np.zeros((128, 128, 3), dtype=np.uint8)


@pytest.fixture
def rng():
    return np.random.default_rng(0)

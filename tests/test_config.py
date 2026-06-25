from gated_sam.config import _coerce, load_config
from gated_sam.seeding import stable_seed


def test_coerce_scalars_and_strings():
    assert _coerce("10") == 10
    assert _coerce("3.5") == 3.5
    assert _coerce("true") is True
    assert _coerce("null") is None
    assert _coerce("vit_h") == "vit_h"
    assert _coerce("/abs/path/to/data") == "/abs/path/to/data"
    assert _coerce("sam_vit_h_4b8939.pth") == "sam_vit_h_4b8939.pth"


def test_coerce_lists():
    # the override that broke Day-1: must become a real list, not the string "[0,1]"
    assert _coerce("[0,1]") == [0, 1]
    assert _coerce("[0,10,20,30]") == [0, 10, 20, 30]


def test_coerce_bareword_lists():
    assert _coerce("[JSRT]") == ["JSRT"]
    assert _coerce("[JSRT,BUSI]") == ["JSRT", "BUSI"]


def test_get_config_rejects_datasets_override():
    from types import SimpleNamespace

    import pytest

    from gated_sam.experiments._common import get_config
    args = SimpleNamespace(config=None, overrides=["datasets=[JSRT]"], datasets=None)
    with pytest.raises(SystemExit):
        get_config(args)


def test_override_list_lands_as_list():
    cfg = load_config(None, ["seeds=[0,1,2]", "noise_levels=[0,30]"])
    assert cfg["seeds"] == [0, 1, 2]
    assert all(isinstance(s, int) for s in cfg["seeds"])
    assert cfg["noise_levels"] == [0, 30]


def test_stable_seed_is_deterministic_and_distinct():
    assert stable_seed("img", 30, 0) == stable_seed("img", 30, 0)
    assert stable_seed("img", 30, 0) != stable_seed("img", 30, 1)

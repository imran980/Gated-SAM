"""Configuration loading. A thin wrapper over a YAML file with dotted-key overrides."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access and dotted get/set, so cfg.search.max_steps works."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        # Wrap nested dicts in place so identity is stable and nested writes persist
        # (cfg.search["guard"] = False must mutate the underlying config, not a copy).
        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, *keys: str) -> Any:
        node: Any = self
        for k in keys:
            node = node[k]
        return node


def _coerce(value: str) -> Any:
    """Best-effort string -> python for CLI overrides.

    Handles ints/floats (10), bools (true/false), null, and literal lists/tuples/dicts
    (`[0,1]`, `[0,10,20,30]`) via ast.literal_eval; anything else (paths, vit_h, file
    names) is left as a string.
    """
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load YAML config and apply `key.sub=value` overrides from the CLI."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = copy.deepcopy(raw)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, value = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return Config(cfg)


def resolve(root: str | Path, sub: str | Path) -> Path:
    """Resolve `sub` against `root` unless `sub` is already absolute."""
    sub = Path(sub)
    return sub if sub.is_absolute() else Path(root) / sub

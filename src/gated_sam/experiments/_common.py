"""Shared CLI / config / model-loading helpers for the experiment scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config, resolve


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to YAML config (default: configs/default.yaml)")
    p.add_argument("--set", nargs="*", default=[], dest="overrides",
                   help="dotted overrides, e.g. --set device=cpu n_images_per_dataset=10 search.max_steps=2")
    p.add_argument("--datasets", nargs="*", default=None, help="subset of dataset names to run")
    return p


def get_config(args):
    cfg = load_config(args.config, args.overrides)
    if args.datasets:
        missing = [d for d in args.datasets if d not in cfg.datasets]
        if missing:
            raise SystemExit(f"unknown datasets: {missing}; available: {list(cfg.datasets)}")
    return cfg


def dataset_names(cfg, args):
    return list(args.datasets) if args.datasets else list(cfg.datasets)


def _require_file(path, what, hint):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at: {path.resolve()}\n"
            f"  hint: {hint}")
    return path


def build_sam(cfg):
    from ..models import load_sam
    ckpt = _require_file(
        resolve(cfg.checkpoint_root, cfg.sam.checkpoint),
        f"SAM checkpoint (model_type={cfg.sam.model_type})",
        "fix checkpoint_root / sam.checkpoint, e.g. "
        "--set checkpoint_root=/abs/path sam.checkpoint=sam_vit_h_4b8939.pth sam.model_type=vit_h")
    return load_sam(cfg.sam.model_type, ckpt, cfg.device)


def build_medsam(cfg):
    from ..models import load_medsam
    ckpt = _require_file(
        resolve(cfg.checkpoint_root, cfg.medsam.checkpoint),
        f"MedSAM checkpoint (model_type={cfg.medsam.model_type})",
        "fix checkpoint_root / medsam.checkpoint, or omit MedSAM")
    return load_medsam(cfg.medsam.model_type, ckpt, cfg.device)


def out_dir(cfg, sub: str) -> Path:
    d = Path(cfg.out_dir) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d

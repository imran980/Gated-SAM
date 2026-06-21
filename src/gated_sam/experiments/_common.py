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


def build_sam(cfg):
    from ..models import load_sam
    return load_sam(cfg.sam.model_type, resolve(cfg.checkpoint_root, cfg.sam.checkpoint), cfg.device)


def build_medsam(cfg):
    from ..models import load_medsam
    return load_medsam(cfg.medsam.model_type, resolve(cfg.checkpoint_root, cfg.medsam.checkpoint), cfg.device)


def out_dir(cfg, sub: str) -> Path:
    d = Path(cfg.out_dir) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d

"""Dataset loaders -> a unified list of Sample(image, gt_mask, gt_box, name, modality).

Lean by design: the reference-free method needs no auxiliary features, so we load only
the image, the ground-truth mask, and its tight box. Path conventions are ported from
the original MICCAI notebook. Each loader is independent; missing datasets are skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import resolve
from .metrics import mask_to_box


@dataclass
class Sample:
    name: str
    image: np.ndarray     # uint8 (S, S, 3)
    gt_mask: np.ndarray   # bool (S, S)
    gt_box: np.ndarray    # [x1, y1, x2, y2]
    modality: str


def _load_rgb(path: Path, size: int) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB").resize((size, size)))


def _load_mask(path: Path, size: int, thresh: int = 0) -> np.ndarray:
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.array(m) > thresh


def _finalize(name, image, gt_mask, modality, size, min_area=100):
    if gt_mask.sum() < min_area:
        return None
    box = mask_to_box(gt_mask, shape=(size, size))
    if box is None:
        return None
    return Sample(name, image, gt_mask, box, modality)


def load_jsrt(spec, data_root, size, n_images):
    img_dir = resolve(data_root, spec["img_dir"])
    mask_dir = resolve(data_root, spec["mask_dir"])
    out = []
    for p in sorted(img_dir.glob("*.jpg"))[:n_images]:
        mp = mask_dir / f"{p.stem}.tif"
        if not mp.exists():
            continue
        s = _finalize(p.stem, _load_rgb(p, size), _load_mask(mp, size, 0), "xray", size)
        if s:
            out.append(s)
    return out


def load_busi(spec, data_root, size, n_images):
    root = resolve(data_root, spec["root"])
    pairs = []
    for folder in ("benign", "malignant"):
        fp = root / folder
        if not fp.exists():
            continue
        for p in fp.glob("*.png"):
            if "_mask" in p.stem:
                continue
            mp = fp / f"{p.stem}_mask.png"
            if mp.exists():
                pairs.append((p, mp))
    rng = np.random.default_rng(42)
    if len(pairs) > n_images:
        pairs = [pairs[i] for i in rng.choice(len(pairs), n_images, replace=False)]
    out = []
    for p, mp in pairs:
        s = _finalize(p.stem, _load_rgb(p, size), _load_mask(mp, size, 0), "ultrasound", size)
        if s:
            out.append(s)
    return out


def load_kvasir(spec, data_root, size, n_images):
    root = resolve(data_root, spec["root"])
    img_dir, mask_dir = root / "images", root / "masks"
    paths = sorted(img_dir.glob("*.jpg"))
    rng = np.random.default_rng(42)
    if len(paths) > n_images:
        paths = [paths[i] for i in rng.choice(len(paths), n_images, replace=False)]
    out = []
    for p in paths:
        mp = mask_dir / p.name
        if not mp.exists():
            continue
        s = _finalize(p.stem, _load_rgb(p, size), _load_mask(mp, size, 127), "endoscopy", size)
        if s:
            out.append(s)
    return out


def load_promise12(spec, data_root, size, n_images):
    import SimpleITK as sitk

    root = resolve(data_root, spec["root"])
    max_slices = int(spec.get("max_slices_per_volume", 5))
    vols = []
    for d in (root / "train_data", root / "test_data", root):
        if not d.exists():
            continue
        for mhd in d.glob("*.mhd"):
            if "_segmentation" in mhd.stem:
                continue
            seg = d / f"{mhd.stem}_segmentation.mhd"
            if seg.exists():
                vols.append((mhd, seg))
    out, per_vol = [], {}
    for mhd, seg in vols:
        if len(out) >= n_images:
            break
        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(mhd)))
        seg_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(seg)))
        for i in range(img_vol.shape[0]):
            if per_vol.get(mhd.stem, 0) >= max_slices or len(out) >= n_images:
                break
            gt = seg_vol[i] > 0
            if gt.sum() < 100:
                continue
            sl = img_vol[i].astype(np.float32)
            sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8) * 255
            img = np.array(Image.fromarray(np.stack([sl.astype(np.uint8)] * 3, -1)).resize((size, size)))
            gtm = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127
            s = _finalize(f"{mhd.stem}_s{i}", img, gtm, "mri", size)
            if s:
                out.append(s)
                per_vol[mhd.stem] = per_vol.get(mhd.stem, 0) + 1
    return out


LOADERS = {"jsrt": load_jsrt, "busi": load_busi, "kvasir": load_kvasir, "promise12": load_promise12}


def load_dataset(name: str, cfg) -> list[Sample]:
    spec = cfg.datasets[name]
    primary = resolve(cfg.data_root, spec.get("root") or spec.get("img_dir"))
    if not primary.exists():
        print(f"  [{name}] WARNING: path not found: {primary.resolve()} "
              f"-> 0 samples (check data_root and folder name)")
        return []
    loader = LOADERS[spec["loader"]]
    samples = loader(spec, cfg.data_root, int(cfg.img_size), int(cfg.n_images_per_dataset))
    print(f"  [{name}] loaded {len(samples)} samples")
    return samples

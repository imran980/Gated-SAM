"""Day 6 — ablations.

Varies one factor of the method at a time and reports Dice at high noise:
  - objective:        predicted_iou | coarse_agreement | perturbation_consistency | combo
  - candidate set:    full vs jitter-only vs no-morphology
  - number of steps:  max_steps in {1, 2, 3}
  - guard:            on vs off  (off should regress on clean prompts -> shows its value)

    python -m gated_sam.experiments.ablations --config configs/default.yaml --set noise_levels=[30]
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..baselines import ours
from ..data import load_dataset
from ..metrics import dice
from ..prompts import add_box_noise
from ..stats import mean_ci
from . import _common


def _variants(cfg):
    """(group, label, cfg-mutation) tuples. Mutations are applied to a deep copy."""
    base = lambda c: c
    V = []
    for o in ["predicted_iou", "coarse_agreement", "perturbation_consistency", "combo"]:
        V.append(("objective", o, lambda c, o=o: c.__setitem__("objective", {**c["objective"], "name": o})))
    V += [
        ("candidates", "full", base),
        ("candidates", "jitter_only", lambda c: c["search"].update(dilate_px=[], erode_px=[])),
        ("candidates", "no_morphology", lambda c: c["search"].update(dilate_px=[], erode_px=[], n_jitter_candidates=8)),
    ]
    for k in (1, 2, 3):
        V.append(("steps", f"max_steps={k}", lambda c, k=k: c["search"].update(max_steps=k)))
    V += [
        ("guard", "guard_on", lambda c: c["search"].update(guard=True)),
        ("guard", "guard_off", lambda c: c["search"].update(guard=False)),
    ]
    return V


def run_variant(sam, samples, cfg, dataset, noise):
    scores = []
    for s in samples:
        sam.set_image(s.image)
        h, w = s.image.shape[:2]
        for seed in cfg.seeds:
            rng = np.random.default_rng((hash(s.name) ^ (noise << 8) ^ seed) % (2**32))
            box = add_box_noise(s.gt_box, int(noise), h, w, rng)
            scores.append(dice(ours(sam, s.image, box, rng, cfg), s.gt_mask))
    return scores


def main(argv=None):
    p = _common.base_parser(__doc__)
    p.add_argument("--noise", type=int, default=30)
    args = p.parse_args(argv)
    cfg = _common.get_config(args)
    out = _common.out_dir(cfg, "ablations")
    sam = _common.build_sam(cfg)

    datasets = _common.dataset_names(cfg, args)
    data = {name: load_dataset(name, cfg) for name in datasets}

    rows = []
    for group, label, mutate in tqdm(_variants(cfg), desc="ablations"):
        vcfg = copy.deepcopy(cfg)
        mutate(vcfg)
        for name in datasets:
            if not data[name]:
                continue
            m, ci = mean_ci(run_variant(sam, data[name], vcfg, name, args.noise))
            rows.append(dict(group=group, variant=label, dataset=name,
                             noise=args.noise, dice_mean=m, dice_ci=ci))

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablations.csv", index=False)
    pivot = df.pivot_table(index=["group", "variant"], columns="dataset", values="dice_mean")
    pivot["AVG"] = pivot.mean(axis=1)
    pivot.to_csv(out / "ablations_pivot.csv")
    print(f"\n=== Ablations (Dice @ δ={args.noise}) ===")
    print(pivot.round(3).to_string())
    print(f"\nWritten to {out}")
    return df


if __name__ == "__main__":
    main()

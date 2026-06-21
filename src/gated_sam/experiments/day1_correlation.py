"""Day 1 — the go/no-go.

For every prediction (one per image x noise x seed) compute the true Dice and three
reference-free signals: SAM predicted-IoU, coarse-mask agreement, and
perturbation-consistency. Then Spearman-correlate each signal against true Dice,
per-dataset and pooled.

GO if perturbation-consistency beats predicted-IoU — especially on BUSI / PROMISE12,
where the predicted-IoU gate currently fails. This produces Figure 2 and is the
empirical proof that the objective the search optimizes is real.

    python -m gated_sam.experiments.day1_correlation --config configs/default.yaml
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from ..data import load_dataset
from ..metrics import dice
from ..objectives import all_signal_objectives
from ..prompts import add_box_noise
from . import _common

SIGNALS = ["predicted_iou", "coarse_agreement", "perturbation_consistency"]


def collect_signals(predictor, samples, cfg, dataset_name):
    """One row per (sample, noise, seed): true Dice + the three signals."""
    objectives = all_signal_objectives(cfg)
    rows = []
    for s in tqdm(samples, desc=f"day1:{dataset_name}"):
        predictor.set_image(s.image)
        h, w = s.image.shape[:2]
        for noise in cfg.noise_levels:
            seeds = cfg.seeds if noise > 0 else [cfg.seeds[0]]   # noise=0 is seed-invariant
            for seed in seeds:
                rng = np.random.default_rng((hash(s.name) ^ (noise << 8) ^ seed) % (2**32))
                box = add_box_noise(s.gt_box, int(noise), h, w, rng)
                pred = predictor.predict_best(box)
                row = {
                    "dataset": dataset_name, "sample": s.name, "noise": noise, "seed": seed,
                    "true_dice": dice(pred.mask, s.gt_mask),
                }
                for name, obj in objectives.items():
                    row[name] = obj(predictor, pred, rng)
                rows.append(row)
    return rows


def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman(signal, true_dice) per dataset and pooled."""
    out = []
    for name, grp in list(df.groupby("dataset")) + [("POOLED", df)]:
        rec = {"dataset": name, "n": len(grp)}
        for sig in SIGNALS:
            rho, p = spearmanr(grp[sig], grp["true_dice"])
            rec[f"{sig}_rho"] = rho
            rec[f"{sig}_p"] = p
        rec["consistency_wins"] = rec["perturbation_consistency_rho"] > rec["predicted_iou_rho"]
        out.append(rec)
    return pd.DataFrame(out)


def verdict(sp: pd.DataFrame) -> str:
    lines = ["", "=" * 64, "DAY-1 GO/NO-GO VERDICT", "=" * 64]
    for _, r in sp.iterrows():
        flag = "GO " if r["consistency_wins"] else "no "
        lines.append(
            f"  [{flag}] {r['dataset']:<10} consistency rho={r['perturbation_consistency_rho']:+.3f} "
            f"vs predicted-IoU rho={r['predicted_iou_rho']:+.3f} "
            f"(coarse={r['coarse_agreement_rho']:+.3f}, n={int(r['n'])})"
        )
    focus = sp[sp["dataset"].isin(["BUSI", "PROMISE12"])]
    pooled = sp[sp["dataset"] == "POOLED"]
    go = bool(pooled["consistency_wins"].all()) and bool(focus["consistency_wins"].all() if len(focus) else True)
    lines.append("-" * 64)
    lines.append("  DECISION: " + ("PROCEED — consistency is the stronger objective."
                                    if go else "STOP / RECONSIDER — consistency did not beat predicted-IoU."))
    lines.append("=" * 64)
    return "\n".join(lines)


def main(argv=None):
    args = _common.base_parser(__doc__).parse_args(argv)
    cfg = _common.get_config(args)
    out = _common.out_dir(cfg, "day1")
    predictor = _common.build_sam(cfg)

    rows = []
    for name in _common.dataset_names(cfg, args):
        samples = load_dataset(name, cfg)
        if samples:
            rows += collect_signals(predictor, samples, cfg, name)

    df = pd.DataFrame(rows)
    df.to_csv(out / "signals.csv", index=False)
    sp = spearman_table(df)
    sp.to_csv(out / "spearman.csv", index=False)
    print("\n" + sp.to_string(index=False))
    print(verdict(sp))

    try:
        from ..figures import day1_correlation_figure
        day1_correlation_figure(df, sp, out)
        print(f"\nFigure 2 + tables written to {out}")
    except Exception as exc:  # plotting must never block the numeric result
        print(f"[warn] figure generation skipped: {exc}")
    return df, sp


if __name__ == "__main__":
    main()

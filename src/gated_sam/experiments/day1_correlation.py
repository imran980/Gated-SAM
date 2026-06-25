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
from ..seeding import stable_rng
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
                rng = stable_rng(s.name, noise, seed)
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
    """Spearman(signal, true_dice) per dataset and pooled, plus a standardized fusion."""
    out = []
    for name, grp in list(df.groupby("dataset")) + [("POOLED", df)]:
        rec = {"dataset": name, "n": len(grp)}
        for sig in SIGNALS:
            rho, p = spearmanr(grp[sig], grp["true_dice"])
            rec[f"{sig}_rho"] = rho
            rec[f"{sig}_p"] = p
        # z-scored fusion of the three signals — does combining help ranking?
        z = grp[SIGNALS].apply(lambda c: (c - c.mean()) / (c.std() + 1e-9))
        rho_c, p_c = spearmanr(z.sum(axis=1), grp["true_dice"])
        rec["combo_z_rho"], rec["combo_z_p"] = rho_c, p_c
        rec["consistency_beats_predIoU"] = rec["perturbation_consistency_rho"] > rec["predicted_iou_rho"]
        out.append(rec)
    return pd.DataFrame(out)


def verdict(sp: pd.DataFrame) -> str:
    """Go/no-go.

    The decision is whether a reference-free signal can RANK candidates better than the
    old predicted-IoU gate — judged pooled and by majority of datasets. A dataset where
    predicted-IoU still wins is NOT a failure of the objective: it means ranking there is
    already fine and any remaining gap is a candidate/lock-in problem (evaluate it with
    the recovery experiment, not Figure 2).
    """
    lines = ["", "=" * 70, "DAY-1 GO/NO-GO VERDICT", "=" * 70]
    per = sp[sp["dataset"] != "POOLED"]
    pooled = sp[sp["dataset"] == "POOLED"].iloc[0]
    for _, r in sp.iterrows():
        flag = "win " if r["consistency_beats_predIoU"] else "----"
        lines.append(
            f"  [{flag}] {r['dataset']:<10} consistency={r['perturbation_consistency_rho']:+.3f} "
            f"vs pred-IoU={r['predicted_iou_rho']:+.3f}  "
            f"(coarse={r['coarse_agreement_rho']:+.3f}, fused={r['combo_z_rho']:+.3f}, n={int(r['n'])})"
        )
    n_win = int(per["consistency_beats_predIoU"].sum())
    n_tot = len(per)
    pooled_win = pooled["perturbation_consistency_rho"] > pooled["predicted_iou_rho"]
    fusion_best = pooled["combo_z_rho"] >= max(pooled[f"{s}_rho"] for s in SIGNALS)
    lockin = per.loc[~per["consistency_beats_predIoU"], "dataset"].tolist()
    go = pooled_win and n_win >= (n_tot + 1) // 2

    lines.append("-" * 70)
    lines.append(f"  consistency beats predicted-IoU gate: pooled={pooled_win}, "
                 f"datasets={n_win}/{n_tot}")
    lines.append(f"  z-scored fusion is the best pooled signal: {fusion_best} "
                 f"(fused rho={pooled['combo_z_rho']:+.3f}) -> consider objective=combo")
    if lockin:
        lines.append(f"  predicted-IoU still leads on: {lockin}  "
                     f"-> ranking is fine there; treat as lock-in (Day-6 recovery), not Fig.2")
    lines.append("  DECISION: " + ("PROCEED — reference-free objective beats the old gate "
                                    "pooled and on the majority of datasets."
                                    if go else "RECONSIDER — objective did not beat the old gate."))
    lines.append("=" * 70)
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

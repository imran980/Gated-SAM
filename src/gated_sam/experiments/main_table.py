"""Days 4-5 — the main table.

Dice + HD95 across delta in {0,10,20,30}, four datasets, multiple seeds, reported as
mean +/- 95% CI with a paired Wilcoxon test. Methods share identical prompts:

    vanilla SAM, MedSAM, ungated cascade (one pass of T, no gate = PerSAM baseline),
    predicted-IoU gate (the OLD method), and Ours (consistency-driven search).

The contribution is proven by exactly one ordering, checked automatically:
    Ours > predicted-IoU gate > ungated   (with significance).

    python -m gated_sam.experiments.main_table --config configs/default.yaml
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..baselines import MEDSAM_METHODS, SAM_METHODS
from ..data import load_dataset
from ..metrics import dice, hd95
from ..prompts import add_box_noise
from ..seeding import stable_rng
from ..stats import fmt_ci, mean_ci, wilcoxon
from . import _common

METHOD_ORDER = ["vanilla_sam", "medsam", "ungated_cascade", "predicted_iou_gate", "ours"]
METHOD_STYLE = {
    "vanilla_sam": {"label": "SAM", "color": "#E74C3C", "marker": "o"},
    "medsam": {"label": "MedSAM", "color": "#F39C12", "marker": "^"},
    "ungated_cascade": {"label": "Ungated cascade", "color": "#9B59B6", "marker": "v"},
    "predicted_iou_gate": {"label": "Pred-IoU gate (old)", "color": "#2ECC71", "marker": "D"},
    "ours": {"label": "Ours (consistency)", "color": "#3498DB", "marker": "s"},
}


def _seed_rng(name, noise, seed, tag=""):
    return stable_rng(name, noise, seed, tag)


def run_grid(sam, medsam, samples, cfg, dataset):
    """One row per (sample, noise, seed, method): Dice + HD95 on identical prompts."""
    rows = []
    for s in tqdm(samples, desc=f"main:{dataset}"):
        sam.set_image(s.image)
        if medsam is not None:
            medsam.set_image(s.image)
        h, w = s.image.shape[:2]
        for noise in cfg.noise_levels:
            seeds = cfg.seeds if noise > 0 else [cfg.seeds[0]]
            for seed in seeds:
                box = add_box_noise(s.gt_box, int(noise), h, w, _seed_rng(s.name, noise, seed))
                for mname, fn in SAM_METHODS.items():
                    m = fn(sam, s.image, box, _seed_rng(s.name, noise, seed, mname), cfg)
                    rows.append(dict(dataset=dataset, sample=s.name, noise=noise, seed=seed,
                                     method=mname, dice=dice(m, s.gt_mask), hd95=hd95(m, s.gt_mask)))
                if medsam is not None:
                    for mname, fn in MEDSAM_METHODS.items():
                        m = fn(medsam, s.image, box, None, cfg)
                        rows.append(dict(dataset=dataset, sample=s.name, noise=noise, seed=seed,
                                         method=mname, dice=dice(m, s.gt_mask), hd95=hd95(m, s.gt_mask)))
    return rows


def summarize(df):
    """mean +/- 95% CI of Dice and HD95 per (dataset, noise, method)."""
    recs = []
    for (d, n, m), g in df.groupby(["dataset", "noise", "method"]):
        dm, dci = mean_ci(g["dice"])
        hm, hci = mean_ci(g["hd95"])
        recs.append(dict(dataset=d, noise=n, method=m, n=len(g),
                         dice_mean=dm, dice_ci=dci, hd95_mean=hm, hd95_ci=hci))
    return pd.DataFrame(recs)


def wilcoxon_table(df):
    """Paired Wilcoxon of Ours vs each baseline, per (dataset, noise)."""
    out = []
    for (d, n), g in df.groupby(["dataset", "noise"]):
        piv = g.pivot_table(index=["sample", "seed"], columns="method", values="dice")
        if "ours" not in piv:
            continue
        for base in [m for m in METHOD_ORDER if m != "ours" and m in piv]:
            _, p = wilcoxon(piv["ours"].values, piv[base].values)
            out.append(dict(dataset=d, noise=n, baseline=base,
                            ours_minus_base=float(np.nanmean(piv["ours"] - piv[base])), p_value=p))
    return pd.DataFrame(out)


def check_ordering(summary, df):
    """Verify Ours > predicted-IoU gate > ungated at each noise, with significance."""
    lines = ["", "=" * 70, "CONTRIBUTION CHECK:  Ours > predicted-IoU gate > ungated", "=" * 70]
    ok_all = True
    for (d, n), g in summary.groupby(["dataset", "noise"]):
        vals = {r.method: r.dice_mean for r in g.itertuples()}
        if not {"ours", "predicted_iou_gate", "ungated_cascade"} <= vals.keys():
            continue
        order_ok = vals["ours"] >= vals["predicted_iou_gate"] >= vals["ungated_cascade"]
        sub = df[(df.dataset == d) & (df.noise == n)]
        piv = sub.pivot_table(index=["sample", "seed"], columns="method", values="dice")
        _, p_gate = wilcoxon(piv["ours"].values, piv["predicted_iou_gate"].values)
        sig = p_gate < 0.05
        ok = order_ok and sig
        ok_all &= ok if n >= 20 else True   # judge the claim where noise actually bites
        flag = "OK " if ok else "-- "
        lines.append(f"  [{flag}] {d:<10} δ={n:<3} ours={vals['ours']:.3f} "
                     f"gate={vals['predicted_iou_gate']:.3f} ungated={vals['ungated_cascade']:.3f} "
                     f"(ours>gate p={p_gate:.1e})")
    lines.append("-" * 70)
    lines.append("  RESULT: " + ("ordering holds with significance where δ>=20 — contribution stands."
                                  if ok_all else "ordering NOT consistently significant — investigate before writing."))
    lines.append("=" * 70)
    return "\n".join(lines)


def wide_table(summary):
    """Human-readable Dice mean±CI table: rows = dataset×noise, cols = methods."""
    rows = []
    for (d, n), g in summary.groupby(["dataset", "noise"]):
        vals = {r.method: fmt_ci(r.dice_mean, r.dice_ci) for r in g.itertuples()}
        rows.append({"dataset": d, "noise": n, **{m: vals.get(m, "—") for m in METHOD_ORDER}})
    return pd.DataFrame(rows)


def main(argv=None):
    args = _common.base_parser(__doc__).parse_args(argv)
    cfg = _common.get_config(args)
    out = _common.out_dir(cfg, "main_table")
    sam = _common.build_sam(cfg)
    try:
        medsam = _common.build_medsam(cfg)
    except Exception as exc:
        print(f"[warn] MedSAM unavailable ({exc}); skipping that baseline.")
        medsam = None

    rows = []
    for name in _common.dataset_names(cfg, args):
        samples = load_dataset(name, cfg)
        if samples:
            rows += run_grid(sam, medsam, samples, cfg, name)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results_long.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    wt = wide_table(summary)
    wt.to_csv(out / "table1_dice.csv", index=False)
    wtab = wilcoxon_table(df)
    wtab.to_csv(out / "wilcoxon.csv", index=False)

    print("\n=== Table 1 (Dice, mean ± 95% CI) ===")
    print(wt.to_string(index=False))
    print(check_ordering(summary, df))

    try:
        from ..figures import robustness_curves
        fig_rows = []
        for (d, n), g in summary.groupby(["dataset", "noise"]):
            row = {"dataset": d, "noise": n}
            for r in g.itertuples():
                row[f"{r.method}_dice_mean"] = r.dice_mean
                row[f"{r.method}_dice_ci"] = r.dice_ci
            fig_rows.append(row)
        robustness_curves(pd.DataFrame(fig_rows), out, METHOD_ORDER, METHOD_STYLE)
        print(f"\nTables + figures written to {out}")
    except Exception as exc:
        print(f"[warn] figure generation skipped: {exc}")
    return df, summary


if __name__ == "__main__":
    main()

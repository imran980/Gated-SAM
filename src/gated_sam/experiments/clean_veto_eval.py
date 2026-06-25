"""Rigorous, oracle-bounded evaluation of the clean_veto decision (no test-set tuning).

`ours` returns EITHER the vanilla mask OR the one anchored refinement, so any decision
rule (signal A-D, any tau, any anchor set) is a pure function of cached per-image records.
We therefore run SAM ONCE (stage `cache`) to record, per (image, noise, seed):

    dice_vanilla, dice_refined(=ungated), dice_search(free search), dice_medsam,
    score0/score1 (SAM predicted-IoU of vanilla/refined),
    q0/q1 (perturbation-consistency of vanilla/refined),
    anchor stats: mask_iou(vanilla,refined), area_ratio, center_shift.

Then stage `analyze` (offline pandas, no SAM) does the τ/signal/anchor sweep on a
deterministic VALIDATION split, picks one global τ, and reports the TEST table once,
plus oracle bound, decision rates, and loss-to-oracle. Stage `qualitative` re-runs SAM
on a handful of selected cases to save overlays.

    python -m gated_sam.experiments.clean_veto_eval --stage cache  --set data_root=... checkpoint_root=...
    python -m gated_sam.experiments.clean_veto_eval --stage analyze
    python -m gated_sam.experiments.clean_veto_eval --stage qualitative --set data_root=... checkpoint_root=...

Decision rules (refine = use the refined mask; else keep vanilla):
    A  initial consistency:        refine iff q0 < tau
    B  delta consistency:          refine iff (q1 - q0) > delta
    C  A + object-anchor ok
    D  B + object-anchor ok
Anchor-ok = mask_iou>=miou_min AND area_ratio in [lo,hi] AND center_shift<=center_max.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data import load_dataset
from ..metrics import dice, iou, mask_to_box
from ..objectives import Objective, build_objective, perturbation_consistency
from ..prompts import add_box_noise, clip_box
from ..refine import refine_search
from ..seeding import stable_rng, stable_seed
from ..stats import fmt_ci, mean_ci
from . import _common

TAUS = [0.75, 0.80, 0.85, 0.90, 0.95, 0.97]
DELTAS = [-0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
MIOU_GRID = [0.50, 0.60, 0.70]
AREA_GRID = [(0.5, 2.0), (0.67, 1.5), (0.75, 1.33)]
VAL_FRACTION = 40           # percent of patients held out for tau selection
REG_BUDGET = 0.015          # max allowed delta=0 Dice regression vs vanilla on validation
METHODS = ["vanilla", "medsam", "ungated", "predicted_iou_gate", "search", "ours", "oracle"]


# ── geometry helpers ─────────────────────────────────────────────────────
def _centroid(mask):
    ys, xs = np.where(mask)
    return None if len(ys) == 0 else np.array([xs.mean(), ys.mean()])


def _center_shift(m0, m1):
    c0, c1 = _centroid(m0), _centroid(m1)
    return 1e9 if c0 is None or c1 is None else float(np.hypot(*(c0 - c1)))


# ── stage 1: cache raw records (the only part that runs SAM) ─────────────
def cache_records(sam, medsam, samples, cfg, dataset, do_search):
    cons = Objective("perturbation_consistency", perturbation_consistency,
                     K=int(cfg.objective.consistency_K), jitter=int(cfg.objective.consistency_jitter))
    rows = []
    for s in tqdm(samples, desc=f"cache:{dataset}"):
        sam.set_image(s.image)
        if medsam is not None:
            medsam.set_image(s.image)
        h, w = s.image.shape[:2]
        pid = s.name.rsplit("_s", 1)[0] if dataset == "PROMISE12" else s.name
        for noise in cfg.noise_levels:
            seeds = cfg.seeds if noise > 0 else [cfg.seeds[0]]
            for seed in seeds:
                rng = stable_rng(s.name, noise, seed)
                box = add_box_noise(s.gt_box, int(noise), h, w, rng)
                p0 = sam.predict_best(clip_box(box, h, w))
                tight = mask_to_box(p0.mask, pad=int(cfg.search.box_pad), shape=(h, w))
                if tight is None:
                    p1 = p0
                else:
                    hint = p0.logits if bool(cfg.search.use_mask_hint) else None
                    p1 = sam.predict_best(tight, mask_input=hint)
                rc = stable_rng(s.name, noise, seed, "cons")
                rec = dict(
                    dataset=dataset, name=s.name, pid=pid, noise=int(noise), seed=int(seed),
                    dice_vanilla=dice(p0.mask, s.gt_mask), dice_refined=dice(p1.mask, s.gt_mask),
                    score0=float(p0.score), score1=float(p1.score),
                    q0=cons(sam, p0, rc), q1=cons(sam, p1, rc),
                    anchor_mask_iou=iou(p0.mask, p1.mask),
                    anchor_area_ratio=p1.mask.sum() / max(int(p0.mask.sum()), 1),
                    anchor_center_shift=_center_shift(p0.mask, p1.mask),
                    dice_medsam=(dice(medsam.predict_best(clip_box(box, h, w)).mask, s.gt_mask)
                                 if medsam is not None else np.nan),
                    dice_search=(dice(refine_search(sam, s.image, box, build_objective(cfg), cfg, rng).mask,
                                      s.gt_mask) if do_search else np.nan),
                )
                rows.append(rec)
    return rows


# ── decision rule (offline) ──────────────────────────────────────────────
def anchor_ok(df, anchors):
    miou_min, lo, hi, center_max = anchors
    return ((df.anchor_mask_iou >= miou_min) &
            df.anchor_area_ratio.between(lo, hi) &
            (df.anchor_center_shift <= center_max)).values


def decide(df, signal, thresh, anchors):
    """Boolean array: True = refine (use refined mask), False = keep vanilla."""
    if signal in ("A", "C"):
        refine = (df.q0 < thresh).values
    else:  # B, D — thresh is a delta on consistency
        refine = ((df.q1 - df.q0) > thresh).values
    if signal in ("C", "D"):
        refine = refine & anchor_ok(df, anchors)
    return refine


def method_dices(df, refine):
    return {
        "vanilla": df.dice_vanilla.values,
        "medsam": df.dice_medsam.values,
        "ungated": df.dice_refined.values,
        "predicted_iou_gate": np.where(df.score1 > df.score0, df.dice_refined, df.dice_vanilla),
        "search": df.dice_search.values,
        "ours": np.where(refine, df.dice_refined, df.dice_vanilla),
        "oracle": np.maximum(df.dice_vanilla, df.dice_refined).values,
    }


# ── tau selection on validation only ─────────────────────────────────────
def assign_split(df):
    df = df.copy()
    df["split"] = np.where(df.pid.map(lambda p: stable_seed(p) % 100 < VAL_FRACTION), "val", "test")
    return df


def tau_sweep(df, signal, anchors):
    """For each tau (A/C) or delta (B/D): val Dice, val delta=0 regression, test Dice."""
    grid = TAUS if signal in ("A", "C") else DELTAS
    val, test = df[df.split == "val"], df[df.split == "test"]
    rows = []
    for t in grid:
        rv = decide(val, signal, t, anchors)
        chosen_v = np.where(rv, val.dice_refined, val.dice_vanilla)
        d0 = (val.noise == 0).values
        reg = float((val.dice_vanilla.values[d0] - chosen_v[d0]).mean()) if d0.any() else 0.0
        rt = decide(test, signal, t, anchors)
        chosen_t = np.where(rt, test.dice_refined, test.dice_vanilla)
        rows.append(dict(signal=signal, thresh=t, val_dice=float(chosen_v.mean()),
                         val_d0_regression=reg, test_dice=float(chosen_t.mean()),
                         refine_rate=float(rt.mean())))
    return pd.DataFrame(rows)


def select_threshold(sweep):
    """Maximize validation Dice subject to the delta=0 regression budget."""
    feas = sweep[sweep.val_d0_regression <= REG_BUDGET]
    pool = feas if len(feas) else sweep
    return float(pool.loc[pool.val_dice.idxmax(), "thresh"])


# ── reporting tables (TEST split) ────────────────────────────────────────
def final_table(df, refine):
    md = method_dices(df, refine)
    rows = []
    for (d, n), idx in df.groupby(["dataset", "noise"]).groups.items():
        pos = df.index.get_indexer(idx)
        row = {"dataset": d, "noise": n}
        for m in METHODS:
            mean, half = mean_ci(md[m][pos])
            row[m] = fmt_ci(mean, half)
        rows.append(row)
    return pd.DataFrame(rows)


def decision_rate_table(df, refine):
    oracle_refine = (df.dice_refined > df.dice_vanilla).values
    out = []
    for (d, n), idx in df.groupby(["dataset", "noise"]).groups.items():
        pos = df.index.get_indexer(idx)
        r, orf = refine[pos], oracle_refine[pos]
        out.append(dict(dataset=d, noise=n, n=len(pos), refine_rate=r.mean(),
                        veto_rate=1 - r.mean(), oracle_agreement=(r == orf).mean()))
    return pd.DataFrame(out)


def loss_to_oracle_table(df, refine):
    md = method_dices(df, refine)
    out = []
    for (d, n), idx in df.groupby(["dataset", "noise"]).groups.items():
        pos = df.index.get_indexer(idx)
        out.append(dict(
            dataset=d, noise=n,
            ours=md["ours"][pos].mean(), oracle=md["oracle"][pos].mean(),
            loss_to_oracle=(md["oracle"][pos] - md["ours"][pos]).mean(),
            vs_vanilla=(md["ours"][pos] - md["vanilla"][pos]).mean(),
            vs_ungated=(md["ours"][pos] - md["ungated"][pos]).mean(),
            vs_gate=(md["ours"][pos] - md["predicted_iou_gate"][pos]).mean(),
        ))
    return pd.DataFrame(out)


def signal_comparison(df):
    """Best threshold per signal A-D (val-selected), reported on test."""
    anchors = (MIOU_GRID[0], *AREA_GRID[0], 0.2 * 256)
    rows = []
    test = df[df.split == "test"]
    for signal in ["A", "B", "C", "D"]:
        sweep = tau_sweep(df, signal, anchors)
        t = select_threshold(sweep)
        r = decide(test, signal, t, anchors)
        md = method_dices(test, r)
        d0 = (test.noise == 0).values
        hi = test.noise.isin([20, 30]).values
        rows.append(dict(
            signal=signal, chosen_thresh=t,
            test_dice=md["ours"].mean(),
            d0_regression=(md["vanilla"][d0] - md["ours"][d0]).mean(),
            hi_gap_vs_ungated=(md["ours"][hi] - md["ungated"][hi]).mean(),
            loss_to_oracle=(md["oracle"] - md["ours"]).mean(),
        ))
    return pd.DataFrame(rows)


def anchor_sweep(df):
    """Vary object-anchor constraints for signal C (best tau per setting)."""
    test = df[df.split == "test"]
    rows = []
    for miou in MIOU_GRID:
        for lo, hi in AREA_GRID:
            anchors = (miou, lo, hi, 0.2 * 256)
            t = select_threshold(tau_sweep(df, "C", anchors))
            r = decide(test, "C", t, anchors)
            md = method_dices(test, r)
            d0 = (test.noise == 0).values
            rows.append(dict(miou_min=miou, area_lo=lo, area_hi=hi, chosen_tau=t,
                             test_dice=md["ours"].mean(), refine_rate=r.mean(),
                             d0_regression=(md["vanilla"][d0] - md["ours"][d0]).mean(),
                             loss_to_oracle=(md["oracle"] - md["ours"]).mean()))
    return pd.DataFrame(rows)


def headline(df, refine):
    md = method_dices(df, refine)
    d0 = (df.noise == 0).values
    hi = df.noise.isin([20, 30]).values
    return {
        "delta0_regression_vs_vanilla": float((md["vanilla"][d0] - md["ours"][d0]).mean()),
        "hi_noise_gap_vs_ungated": float((md["ours"][hi] - md["ungated"][hi]).mean()),
        "avg_loss_to_oracle": float((md["oracle"] - md["ours"]).mean()),
        "avg_gain_vs_vanilla": float((md["ours"] - md["vanilla"]).mean()),
        "avg_gain_vs_gate": float((md["ours"] - md["predicted_iou_gate"]).mean()),
    }


# ── stages ───────────────────────────────────────────────────────────────
def _records_path(cfg):
    return _common.out_dir(cfg, "clean_veto") / "records.csv"


def run_cache(cfg, args):
    sam = _common.build_sam(cfg)
    try:
        medsam = _common.build_medsam(cfg)
    except Exception as exc:
        print(f"[warn] MedSAM unavailable ({exc}); dice_medsam=NaN")
        medsam = None
    rows = []
    for name in _common.dataset_names(cfg, args):
        samples = load_dataset(name, cfg)
        if samples:
            rows += cache_records(sam, medsam, samples, cfg, name, do_search=not args.skip_search)
    df = pd.DataFrame(rows)
    df.to_csv(_records_path(cfg), index=False)
    print(f"\ncached {len(df)} records -> {_records_path(cfg)}")
    return df


def run_analyze(cfg, args):
    out = _common.out_dir(cfg, "clean_veto")
    df = assign_split(pd.read_csv(_records_path(cfg))).reset_index(drop=True)
    anchors_default = (MIOU_GRID[0], *AREA_GRID[0], 0.2 * int(cfg.img_size))

    sweep = tau_sweep(df, args.signal, anchors_default)
    tau = select_threshold(sweep)
    test = df[df.split == "test"].reset_index(drop=True)
    refine_test = decide(test, args.signal, tau, anchors_default)

    tables = {
        "tau_sweep": sweep,
        "table1_test": final_table(test, refine_test),
        "oracle_decision_rates": decision_rate_table(test, refine_test),
        "loss_to_oracle": loss_to_oracle_table(test, refine_test),
        "signal_comparison": signal_comparison(df),
        "anchor_sweep": anchor_sweep(df),
    }
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)

    print(f"\nSIGNAL={args.signal}  selected threshold={tau}  "
          f"(val-selected, reg budget<= {REG_BUDGET}; val/test split {VAL_FRACTION}/{100-VAL_FRACTION} by patient)")
    print("\n=== tau / delta sweep (val-selected, no test peeking) ===")
    print(sweep.round(4).to_string(index=False))
    print("\n=== FINAL Table 1 — TEST split (Dice mean ± 95% CI) ===")
    print(tables["table1_test"].to_string(index=False))
    print("\n=== Oracle bound + decision rates (TEST) ===")
    print(tables["oracle_decision_rates"].round(3).to_string(index=False))
    print("\n=== Loss-to-oracle / gains (TEST) ===")
    print(tables["loss_to_oracle"].round(3).to_string(index=False))
    print("\n=== Veto-signal comparison A/B/C/D (TEST, each val-selected) ===")
    print(tables["signal_comparison"].round(4).to_string(index=False))
    print("\n=== Object-anchor sweep, signal C (TEST) ===")
    print(tables["anchor_sweep"].round(4).to_string(index=False))

    hl = headline(test, refine_test)
    print("\n" + "=" * 70)
    print("HEADLINE (the claim is 'protect clean prompts, keep most high-noise benefit'):")
    print(f"  delta=0 regression vs vanilla : {hl['delta0_regression_vs_vanilla']:+.3f}  (want ~0)")
    print(f"  delta>=20 gap vs ungated      : {hl['hi_noise_gap_vs_ungated']:+.3f}  (want >= ~0)")
    print(f"  avg loss to oracle            : {hl['avg_loss_to_oracle']:.3f}  (want small)")
    print(f"  avg gain vs predicted-IoU gate: {hl['avg_gain_vs_gate']:+.3f}")
    print("=" * 70)
    print(f"\nAll tables written to {out}")
    return tables


def run_qualitative(cfg, args):
    """Re-run SAM on a few selected cases and save overlays (5 archetypes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _common.out_dir(cfg, "clean_veto")
    df = assign_split(pd.read_csv(_records_path(cfg))).reset_index(drop=True)
    anchors = (MIOU_GRID[0], *AREA_GRID[0], 0.2 * int(cfg.img_size))
    df["refine"] = decide(df, args.signal, args.tau, anchors)
    df["oracle_refine"] = df.dice_refined > df.dice_vanilla

    picks = {
        "clean_correctly_vetoed": df[(df.noise == 0) & (~df.refine) & (~df.oracle_refine)],
        "noisy_correctly_refined": df[(df.noise >= 20) & (df.refine) & (df.dice_refined > df.dice_vanilla + 0.1)],
        "noisy_wrongly_vetoed": df[(df.noise >= 20) & (~df.refine) & (df.dice_refined > df.dice_vanilla + 0.1)],
        "clean_wrongly_refined": df[(df.noise == 0) & (df.refine) & (df.dice_refined < df.dice_vanilla - 0.05)],
        "goodhart_search_failure": df[(df.dice_search < df.dice_vanilla - 0.2)],
    }
    sam = _common.build_sam(cfg)
    sample_cache = {name: {s.name: s for s in load_dataset(name, cfg)}
                    for name in _common.dataset_names(cfg, args)}

    for label, sub in picks.items():
        if not len(sub):
            print(f"[qual] no example for {label}")
            continue
        r = sub.sort_values("dice_vanilla").iloc[len(sub) // 2]   # a median, not a cherry-pick
        s = sample_cache.get(r.dataset, {}).get(r["name"])
        if s is None:
            continue
        h, w = s.image.shape[:2]
        rng = stable_rng(r["name"], r.noise, r.seed)
        box = add_box_noise(s.gt_box, int(r.noise), h, w, rng)
        sam.set_image(s.image)
        p0 = sam.predict_best(clip_box(box, h, w))
        tight = mask_to_box(p0.mask, pad=int(cfg.search.box_pad), shape=(h, w))
        p1 = sam.predict_best(tight, mask_input=p0.logits) if tight is not None else p0
        srch = refine_search(sam, s.image, box, build_objective(cfg), cfg, rng).mask
        panels = [("image", None), ("GT", s.gt_mask), (f"vanilla {r.dice_vanilla:.2f}", p0.mask),
                  (f"refined {r.dice_refined:.2f}", p1.mask), (f"search {r.dice_search:.2f}", srch)]
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
        for ax, (title, mask) in zip(axes, panels):
            ax.imshow(s.image)
            if mask is not None:
                ax.imshow(np.ma.masked_where(~mask.astype(bool), mask), alpha=0.45, cmap="autumn")
            ax.set_title(title, fontsize=9); ax.axis("off")
        fig.suptitle(f"{label} | {r.dataset} {r['name']} δ={r.noise}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"qual_{label}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[qual] saved qual_{label}.png ({r.dataset} {r['name']} δ={r.noise})")
    print(f"\nQualitative panels written to {out}")


def main(argv=None):
    p = _common.base_parser(__doc__)
    p.add_argument("--stage", choices=["cache", "analyze", "qualitative"], required=True)
    p.add_argument("--signal", choices=["A", "B", "C", "D"], default="A")
    p.add_argument("--tau", type=float, default=0.90, help="qualitative: threshold for the decision")
    p.add_argument("--skip-search", action="store_true", help="cache: skip the slow free-search column")
    args = p.parse_args(argv)
    cfg = _common.get_config(args)
    if args.stage == "cache":
        run_cache(cfg, args)
    elif args.stage == "analyze":
        run_analyze(cfg, args)
    else:
        run_qualitative(cfg, args)


if __name__ == "__main__":
    main()

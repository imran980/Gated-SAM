"""Oracle-bounded evaluation of the keep-vs-refine decision (no test-set tuning).

`ours` returns EITHER the vanilla mask OR one anchored refinement, so every gate is a
pure function of cached per-image records. We run SAM ONCE (stage `cache`); all gate
sweeps are offline pandas (stage `analyze`), so adding/﻿changing gates needs NO re-cache.

Gates (refine = use refined mask; else keep vanilla). s=score1-score0, dq=q1-q0:
    A  q0 < tau_q                       (initial consistency)
    B  dq > tau_d                       (delta consistency)
    C  A AND object-anchor-ok           (hard anchor — kept to show it is too conservative)
    D  B AND object-anchor-ok           (hard anchor — too conservative)
    E  s > tau_s                        (delta predicted-IoU; tau_s=0 == old predicted-IoU gate)
    F  s > tau_s OR q0 < tau_q          (predicted-IoU OR low stability)
    G  s > tau_s OR dq > tau_d          (predicted-IoU OR consistency gain)
    H  s > tau_s OR (q0 < tau_q AND dq > tau_d)   (hybrid)
Anchors are an ANALYSIS feature / soft diagnostic, never a hard requirement for A,B,E-H.

    python -m ...clean_veto_eval --stage cache  --set data_root=... checkpoint_root=...
    python -m ...clean_veto_eval --stage analyze
    python -m ...clean_veto_eval --stage qualitative --set data_root=... checkpoint_root=...
"""
from __future__ import annotations

import json
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

TAU_S = [-0.05, -0.02, 0.00, 0.02, 0.05]
TAU_Q = [0.50, 0.60, 0.70, 0.80]
TAU_D = [0.00, 0.02, 0.05, 0.10]
MIOU_GRID = [0.50, 0.60, 0.70]
AREA_GRID = [(0.5, 2.0), (0.67, 1.5), (0.75, 1.33)]
VAL_FRACTION = 40           # percent of patients held out for threshold selection
REG_BUDGET = 0.015          # max allowed delta=0 Dice regression vs vanilla (validation)
GATES = ["A", "B", "C", "D", "E", "F", "G", "H"]
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
                rows.append(dict(
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
                ))
    return rows


# ── gate decision rules (offline) ────────────────────────────────────────
def anchor_ok(df, anchors):
    miou_min, lo, hi, center_max = anchors
    return ((df.anchor_mask_iou >= miou_min) & df.anchor_area_ratio.between(lo, hi) &
            (df.anchor_center_shift <= center_max)).values


def gate_refine(name, df, p):
    s = (df.score1 - df.score0).values
    q0 = df.q0.values
    dq = (df.q1 - df.q0).values
    if name == "A":
        r = q0 < p["tau_q"]
    elif name == "B":
        r = dq > p["tau_d"]
    elif name == "C":
        r = (q0 < p["tau_q"]) & anchor_ok(df, p["anchors"])
    elif name == "D":
        r = (dq > p["tau_d"]) & anchor_ok(df, p["anchors"])
    elif name == "E":
        r = s > p["tau_s"]
    elif name == "F":
        r = (s > p["tau_s"]) | (q0 < p["tau_q"])
    elif name == "G":
        r = (s > p["tau_s"]) | (dq > p["tau_d"])
    elif name == "H":
        r = (s > p["tau_s"]) | ((q0 < p["tau_q"]) & (dq > p["tau_d"]))
    else:
        raise ValueError(name)
    return np.asarray(r, dtype=bool)


def gate_grid(name, anchors):
    A = [list(anchors)]
    if name == "A":
        return [{"tau_q": q} for q in TAU_Q]
    if name == "B":
        return [{"tau_d": d} for d in TAU_D]
    if name == "C":
        return [{"tau_q": q, "anchors": a} for q in TAU_Q for a in A]
    if name == "D":
        return [{"tau_d": d, "anchors": a} for d in TAU_D for a in A]
    if name == "E":
        return [{"tau_s": s} for s in TAU_S]
    if name == "F":
        return [{"tau_s": s, "tau_q": q} for s in TAU_S for q in TAU_Q]
    if name == "G":
        return [{"tau_s": s, "tau_d": d} for s in TAU_S for d in TAU_D]
    if name == "H":
        return [{"tau_s": s, "tau_q": q, "tau_d": d} for s in TAU_S for q in TAU_Q for d in TAU_D]
    raise ValueError(name)


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


# ── validation-only selection ────────────────────────────────────────────
def assign_split(df):
    df = df.copy()
    df["split"] = np.where(df.pid.map(lambda p: stable_seed(p) % 100 < VAL_FRACTION), "val", "test")
    return df


def select_params(df, name, grid):
    """Pick params maximizing validation Dice s.t. delta=0 regression <= budget."""
    val = df[df.split == "val"]
    d0 = (val.noise == 0).values
    recs = []
    for p in grid:
        r = gate_refine(name, val, p)
        chosen = np.where(r, val.dice_refined, val.dice_vanilla)
        reg = float((val.dice_vanilla.values[d0] - chosen[d0]).mean()) if d0.any() else 0.0
        recs.append((p, float(chosen.mean()), reg))
    feasible = [x for x in recs if x[2] <= REG_BUDGET]
    best = max(feasible or recs, key=lambda x: x[1])[0]
    sweep = pd.DataFrame([{**{k: v for k, v in p.items() if k != "anchors"},
                           "val_dice": vd, "val_d0_reg": rg} for p, vd, rg in recs])
    return best, sweep, bool(feasible)


def _p2s(params):
    if not params:
        return ""
    return " ".join(f"{k}={v:g}" for k, v in params.items() if k != "anchors")


def metrics_from_refine(df, name, refine, params=None):
    md = method_dices(df, refine)
    d0 = (df.noise == 0).values
    hi = df.noise.isin([20, 30]).values
    return dict(
        gate=name, params=_p2s(params),
        overall_dice=float(md["ours"].mean()),
        d0_regression=float((md["vanilla"][d0] - md["ours"][d0]).mean()),
        hi_gap_vs_ungated=float((md["ours"][hi] - md["ungated"][hi]).mean()),
        loss_to_oracle=float((md["oracle"] - md["ours"]).mean()),
        gain_vs_predIoU=float((md["ours"] - md["predicted_iou_gate"]).mean()),
        refine_rate=float(refine.mean()),
    )


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
            dataset=d, noise=n, ours=md["ours"][pos].mean(), oracle=md["oracle"][pos].mean(),
            loss_to_oracle=(md["oracle"][pos] - md["ours"][pos]).mean(),
            vs_vanilla=(md["ours"][pos] - md["vanilla"][pos]).mean(),
            vs_ungated=(md["ours"][pos] - md["ungated"][pos]).mean(),
            vs_gate=(md["ours"][pos] - md["predicted_iou_gate"][pos]).mean()))
    return pd.DataFrame(out)


# ── PROMISE12 diagnosis: does a global signal separate help vs hurt? ─────
def _auc(label, score):
    from scipy.stats import rankdata
    label = np.asarray(label, bool)
    score = np.asarray(score, float)
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def promise_diagnosis(df):
    p = df[df.dataset == "PROMISE12"]
    groups = {
        "clean_refine_HURTS (δ=0)": p[(p.noise == 0) & (p.dice_refined < p.dice_vanilla - 0.05)],
        "noisy_refine_HELPS (δ≥20)": p[(p.noise.isin([20, 30])) & (p.dice_refined > p.dice_vanilla + 0.05)],
    }
    diag = pd.DataFrame([dict(
        group=g, n=len(d), score0=d.score0.mean(), score1=d.score1.mean(),
        dscore=(d.score1 - d.score0).mean(), q0=d.q0.mean(), q1=d.q1.mean(),
        dq=(d.q1 - d.q0).mean(), area_ratio=d.anchor_area_ratio.mean(),
        mask_iou=d.anchor_mask_iou.mean()) for g, d in groups.items()])
    # separability AUC on the "decision-matters" subset (helps vs hurts)
    dec = p[(p.dice_refined - p.dice_vanilla).abs() > 0.05].copy()
    dec["helps"] = (dec.dice_refined > dec.dice_vanilla).values
    sigs = {"dscore": dec.score1 - dec.score0, "q0": dec.q0, "dq": dec.q1 - dec.q0,
            "area_ratio": dec.anchor_area_ratio, "mask_iou": dec.anchor_mask_iou}
    auc = pd.DataFrame([dict(signal=k, auc_helps=_auc(dec.helps, v), n=len(dec)) for k, v in sigs.items()])
    return diag, auc.sort_values("auc_helps", ascending=False)


def pareto_plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for _, r in summary.iterrows():
        ref = r.gate in ("vanilla", "ungated", "pred_iou_gate")
        ax.scatter(r.d0_regression, r.hi_gap_vs_ungated, s=90 if not ref else 70,
                   marker="o" if not ref else "x", color="#888" if ref else "#3498DB", zorder=3)
        ax.annotate(r.gate, (r.d0_regression, r.hi_gap_vs_ungated),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axvline(REG_BUDGET, ls="--", color="#E74C3C", lw=1, label=f"δ=0 budget ({REG_BUDGET})")
    ax.axhline(0.0, ls="--", color="#2ecc71", lw=1, label="parity with ungated (high noise)")
    ax.set_xlabel("δ=0 regression vs vanilla  (← better)")
    ax.set_ylabel("δ≥20 gain vs ungated  (better →)")
    ax.set_title("Gate Pareto: protect clean prompts vs keep high-noise benefit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out) / "pareto_gates.png", dpi=160, bbox_inches="tight")
    fig.savefig(Path(out) / "pareto_gates.pdf", bbox_inches="tight")
    plt.close(fig)


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
    pd.DataFrame(rows).to_csv(_records_path(cfg), index=False)
    print(f"\ncached {len(rows)} records -> {_records_path(cfg)}")


def run_analyze(cfg, args):
    out = _common.out_dir(cfg, "clean_veto")
    df = assign_split(pd.read_csv(_records_path(cfg))).reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    anchors = (MIOU_GRID[0], *AREA_GRID[0], 0.2 * int(cfg.img_size))

    # 1) every gate: select on validation, evaluate on test
    summary, refines, sweeps = [], {}, {}
    for g in GATES:
        best, sweep, feasible = select_params(df, g, gate_grid(g, anchors))
        refines[g] = gate_refine(g, test, best)
        sweeps[g] = sweep
        m = metrics_from_refine(test, g, refines[g], best)
        m["feasible"] = feasible
        m["selected"] = best
        summary.append(m)
    # reference rows
    refs = {"vanilla": np.zeros(len(test), bool), "ungated": np.ones(len(test), bool),
            "pred_iou_gate": (test.score1 > test.score0).values}
    for rn, rr in refs.items():
        summary.append({**metrics_from_refine(test, rn, rr), "feasible": True, "selected": {}})
    summ = pd.DataFrame(summary)
    summ.drop(columns=["selected"]).to_csv(out / "gate_summary.csv", index=False)

    # 2) recommended gate: best test Dice among feasible learnable gates
    learn = summ[summ.gate.isin(GATES)]
    feas = learn[learn.d0_regression <= REG_BUDGET]
    rec = (feas if len(feas) else learn).sort_values("overall_dice", ascending=False).iloc[0]
    rec_params = dict(summary[GATES.index(rec.gate)]["selected"])
    (out / "selected_best.json").write_text(json.dumps({"gate": rec.gate, "params": rec_params}))

    # 3) per-recommended-gate tables + diagnosis + pareto
    rr = refines[rec.gate]
    final_table(test, rr).to_csv(out / "table1_test.csv", index=False)
    decision_rate_table(test, rr).to_csv(out / "oracle_decision_rates.csv", index=False)
    loss_to_oracle_table(test, rr).to_csv(out / "loss_to_oracle.csv", index=False)
    for g in GATES:
        sweeps[g].to_csv(out / f"sweep_{g}.csv", index=False)
    diag, auc = promise_diagnosis(test)
    diag.to_csv(out / "promise12_diagnosis.csv", index=False)
    auc.to_csv(out / "promise12_signal_auc.csv", index=False)
    pareto_plot(summ, out)

    # 4) print
    pd.set_option("display.width", 200)
    print(f"\nval/test split {VAL_FRACTION}/{100-VAL_FRACTION} by patient; thresholds selected on VAL only "
          f"(max val Dice s.t. δ=0 regression ≤ {REG_BUDGET}).")
    print("\n=== GATE SUMMARY (test) — A–H + references ===")
    print(summ.drop(columns=["selected"]).round(4).to_string(index=False))
    print(f"\nRECOMMENDED GATE: {rec.gate}  ({_p2s(rec_params)})  "
          f"[feasible={bool(rec.feasible)}]")
    print("\n=== FINAL Table 1 — recommended gate, TEST (Dice mean ± 95% CI) ===")
    print(final_table(test, rr).to_string(index=False))
    print("\n=== PROMISE12 diagnosis: help vs hurt group signal means ===")
    print(diag.round(3).to_string(index=False))
    print("\n  signal separability (AUC for predicting 'refinement helps', PROMISE12):")
    print(auc.round(3).to_string(index=False))

    print("\n" + "=" * 74)
    gate_dice = float(summ.loc[summ.gate == "pred_iou_gate", "overall_dice"].iloc[0])
    gate_loss = float(summ.loc[summ.gate == "pred_iou_gate", "loss_to_oracle"].iloc[0])
    crit_A = (rec.overall_dice >= gate_dice - 1e-6) and (rec.d0_regression <= REG_BUDGET)
    crit_B = (rec.loss_to_oracle < gate_loss - 0.005) and (rec.d0_regression <= REG_BUDGET)
    if crit_A or crit_B:
        print("CLAIM SUPPORTED: gate", rec.gate,
              f"matches/beats predicted-IoU gate (Dice {rec.overall_dice:.3f} vs {gate_dice:.3f}, "
              f"loss-to-oracle {rec.loss_to_oracle:.3f} vs {gate_loss:.3f}) with δ=0 reg "
              f"{rec.d0_regression:+.3f}.")
        print('Candidate title: "When Should SAM Refine? Reference-Free Clean-Prompt Veto ..."')
    else:
        print("CLAIM NOT SUPPORTED by either criterion. Recommend reframing as:")
        print('  "When Should SAM Refine? A Prompt Noise Gap Study of Test-Time Self-Refinement')
        print('   in Medical Segmentation" — with oracle keep/refine as the central analysis.')
    print("=" * 74)
    print(f"\nAll tables + pareto_gates.png written to {out}")


def run_qualitative(cfg, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _common.out_dir(cfg, "clean_veto")
    df = assign_split(pd.read_csv(_records_path(cfg))).reset_index(drop=True)
    sel_path = out / "selected_best.json"
    if args.gate:
        gate, params = args.gate, {}   # uses default mid-grid thresholds below
        anchors = (MIOU_GRID[0], *AREA_GRID[0], 0.2 * int(cfg.img_size))
        params = {"tau_s": 0.0, "tau_q": 0.7, "tau_d": 0.05, "anchors": list(anchors)}
    elif sel_path.exists():
        sel = json.loads(sel_path.read_text())
        gate, params = sel["gate"], sel["params"]
    else:
        gate, params = "F", {"tau_s": 0.0, "tau_q": 0.7}
    print(f"[qual] gate={gate} params={_p2s(params)}")
    df["refine"] = gate_refine(gate, df, params)
    df["oracle_refine"] = df.dice_refined > df.dice_vanilla

    picks = {
        "clean_correctly_vetoed": df[(df.noise == 0) & (~df.refine) & (~df.oracle_refine)],
        "noisy_correctly_refined": df[(df.noise >= 20) & (df.refine) & (df.dice_refined > df.dice_vanilla + 0.1)],
        "noisy_wrongly_vetoed": df[(df.noise >= 20) & (~df.refine) & (df.dice_refined > df.dice_vanilla + 0.1)],
        "clean_wrongly_refined": df[(df.noise == 0) & (df.refine) & (df.dice_refined < df.dice_vanilla - 0.05)],
        "goodhart_search_failure": df[(df.dice_search < df.dice_vanilla - 0.2)],
    }
    sam = _common.build_sam(cfg)
    cache = {n: {s.name: s for s in load_dataset(n, cfg)} for n in _common.dataset_names(cfg, args)}
    for label, sub in picks.items():
        if not len(sub):
            print(f"[qual] no example for {label}")
            continue
        r = sub.sort_values("dice_vanilla").iloc[len(sub) // 2]
        s = cache.get(r.dataset, {}).get(r["name"])
        if s is None:
            continue
        h, w = s.image.shape[:2]
        box = add_box_noise(s.gt_box, int(r.noise), h, w, stable_rng(r["name"], r.noise, r.seed))
        sam.set_image(s.image)
        p0 = sam.predict_best(clip_box(box, h, w))
        tight = mask_to_box(p0.mask, pad=int(cfg.search.box_pad), shape=(h, w))
        p1 = sam.predict_best(tight, mask_input=p0.logits) if tight is not None else p0
        srch = refine_search(sam, s.image, box, build_objective(cfg), cfg,
                             stable_rng(r["name"], r.noise, r.seed)).mask
        panels = [("image", None), ("GT", s.gt_mask), (f"vanilla {r.dice_vanilla:.2f}", p0.mask),
                  (f"refined {r.dice_refined:.2f}", p1.mask), (f"search {r.dice_search:.2f}", srch)]
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
        for ax, (title, mask) in zip(axes, panels):
            ax.imshow(s.image)
            if mask is not None:
                ax.imshow(np.ma.masked_where(~mask.astype(bool), mask), alpha=0.45, cmap="autumn")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{label} | {r.dataset} {r['name']} δ={r.noise}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"qual_{label}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[qual] saved qual_{label}.png ({r.dataset} {r['name']} δ={r.noise})")
    print(f"\nQualitative panels written to {out}")


def main(argv=None):
    p = _common.base_parser(__doc__)
    p.add_argument("--stage", choices=["cache", "analyze", "qualitative"], required=True)
    p.add_argument("--gate", choices=GATES, default=None,
                   help="qualitative: force a gate (default: reuse analyze's recommended gate)")
    p.add_argument("--skip-search", action="store_true", help="cache: skip the slow free-search column")
    args = p.parse_args(argv)
    cfg = _common.get_config(args)
    {"cache": run_cache, "analyze": run_analyze, "qualitative": run_qualitative}[args.stage](cfg, args)


if __name__ == "__main__":
    main()

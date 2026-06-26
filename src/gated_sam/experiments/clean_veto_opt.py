"""Final VAL-only optimization pass over the refinement candidate and the keep/refine gate.

Improves gate I by jointly optimizing, on the validation split only:
  - refinement box padding (pad_ratio),
  - SAM multimask selection via a reference-free utility U = score + l1*consistency - l2*area_penalty,
  - the gate: I (override), J (calibrated utility), or K (predicted-IoU base + override).
Selection maximizes mean(Dice+IoU)/2 subject to delta=0 Dice AND IoU regression <= 0.015.
TEST is evaluated exactly once with the chosen config. Reports Dice and IoU throughout.

This needs a fresh, richer cache (pads x multimask, Dice+IoU, +postproc variant):
    python -m gated_sam.experiments.clean_veto_opt --stage cache --set data_root=... checkpoint_root=...
    python -m gated_sam.experiments.clean_veto_opt --stage analyze            # offline, no SAM
    python -m gated_sam.experiments.clean_veto_opt --stage analyze --postproc # same, with CC+hole-fill on ALL methods
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

PAD_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]
LAM1 = [0.0, 0.25, 0.5, 1.0]
LAM2 = [0.0, 0.25, 0.5, 1.0]
VAL_FRACTION = 40
REG_BUDGET = 0.015
TABLE_METHODS = ["vanilla", "medsam", "ungated", "predicted_iou_gate", "search", "ours", "oracle"]

GRID_I = [{"R": R, "tau_d": td, "tau_s": ts, "tau_bad": tb, "R_bad": rb}
          for R in [2.0, 3.0, 4.0, 5.0] for td in [0.05, 0.10, 0.15, 0.20]
          for ts in [-0.05, -0.02, 0.0] for tb in [-0.08, -0.05, -0.02] for rb in [1.5, 2.0, 2.5]]
GRID_K = [{"R": R, "tau_d": td, "tau_s": ts} for R in [1.5, 2.0, 2.5, 3.0, 4.0]
          for td in [0.02, 0.05, 0.08, 0.10, 0.15] for ts in [-0.05, -0.02, 0.0, 0.02]]
GRID_J = [{"a": a, "b": b, "c": c, "d": d, "thr": thr, "tau_bad": tb, "R_bad": rb}
          for a in [0.5, 1.0, 2.0] for b in [0.5, 1.0, 2.0] for c in [0.1, 0.25, 0.5]
          for d in [0.5, 1.0] for thr in [-0.05, 0.0, 0.05, 0.10]
          for tb in [-0.08, -0.05, -0.02] for rb in [1.5, 2.0]]


# ── postprocessing (applied identically to all methods when enabled) ─────
def postproc(mask):
    from scipy.ndimage import binary_fill_holes, label

    m = mask.astype(bool)
    if m.sum() == 0:
        return m
    lab, n = label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == sizes.argmax()
    return binary_fill_holes(m)


def _expand(box, ratio, h, w):
    x1, y1, x2, y2 = (float(v) for v in box)
    pw, ph = ratio * (x2 - x1), ratio * (y2 - y1)
    return clip_box(np.array([x1 - pw, y1 - ph, x2 + pw, y2 + ph]), h, w)


def _di(mask, gt):
    return dice(mask, gt), iou(mask, gt)


# ── stage cache (the only SAM-bound part) ────────────────────────────────
def cache(sam, medsam, samples, cfg, dataset, do_search):
    cons = Objective("perturbation_consistency", perturbation_consistency,
                     K=int(cfg.objective.consistency_K), jitter=int(cfg.objective.consistency_jitter))
    base, cand = [], []
    for s in tqdm(samples, desc=f"cache:{dataset}"):
        sam.set_image(s.image)
        if medsam is not None:
            medsam.set_image(s.image)
        h, w = s.image.shape[:2]
        gt = s.gt_mask
        pid = s.name.rsplit("_s", 1)[0] if dataset == "PROMISE12" else s.name
        for noise in cfg.noise_levels:
            for seed in (cfg.seeds if noise > 0 else [cfg.seeds[0]]):
                rng = stable_rng(s.name, noise, seed)
                box = clip_box(add_box_noise(s.gt_box, int(noise), h, w, rng), h, w)
                p0 = sam.predict_best(box)
                q0 = cons(sam, p0, stable_rng(s.name, noise, seed, "c0"))
                dv, iv = _di(p0.mask, gt)
                dvp, ivp = _di(postproc(p0.mask), gt)
                row = dict(dataset=dataset, name=s.name, pid=pid, noise=int(noise), seed=int(seed),
                           dice_vanilla=dv, iou_vanilla=iv, dice_vanilla_pp=dvp, iou_vanilla_pp=ivp,
                           score0=float(p0.score), q0=float(q0))
                for tag, pred in [("medsam", medsam.predict_best(box) if medsam is not None else None),
                                  ("search", None)]:
                    if tag == "medsam" and pred is not None:
                        d, i = _di(pred.mask, gt); dp, ip = _di(postproc(pred.mask), gt)
                    elif tag == "search" and do_search:
                        m = refine_search(sam, s.image, box, build_objective(cfg), cfg, rng).mask
                        d, i = _di(m, gt); dp, ip = _di(postproc(m), gt)
                    else:
                        d = i = dp = ip = np.nan
                    row.update({f"dice_{tag}": d, f"iou_{tag}": i,
                                f"dice_{tag}_pp": dp, f"iou_{tag}_pp": ip})
                base.append(row)

                tight = mask_to_box(p0.mask, pad=0, shape=(h, w))
                if tight is None:
                    continue
                a0 = max(int(p0.mask.sum()), 1)
                for pad in PAD_GRID:
                    rbox = _expand(tight, pad, h, w)
                    hint = p0.logits if bool(cfg.search.use_mask_hint) else None
                    preds = sam.predict_all(rbox, mask_input=hint)
                    qbox = float(cons(sam, preds[0], stable_rng(s.name, noise, seed, "c1", str(pad))))
                    for ch, pr in enumerate(preds):
                        d, i = _di(pr.mask, gt); dp, ip = _di(postproc(pr.mask), gt)
                        cand.append(dict(name=s.name, noise=int(noise), seed=int(seed),
                                         pad=float(pad), channel=ch, score=float(pr.score), q_box=qbox,
                                         area_ratio=pr.mask.sum() / a0, mask_iou_vanilla=iou(pr.mask, p0.mask),
                                         dice=d, iou=i, dice_pp=dp, iou_pp=ip))
    return base, cand


# ── offline machinery ────────────────────────────────────────────────────
def select_refined(cand, pad, l1, l2):
    sub = cand[cand["pad"] == pad]   # bracket: 'pad' collides with DataFrame.pad()
    areapen = np.log(np.clip(sub.area_ratio.values, 1e-6, None)) ** 2   # symmetric extreme-area penalty
    sub = sub.assign(U=sub.score.values + l1 * sub.q_box.values - l2 * areapen)
    sel = sub.loc[sub.groupby(["name", "noise", "seed"]).U.idxmax()]
    return sel.rename(columns={"score": "score1", "q_box": "q1", "dice": "dice_refined",
                               "iou": "iou_refined", "dice_pp": "dice_refined_pp",
                               "iou_pp": "iou_refined_pp"})


def build_frame(base, cand, pad, l1, l2):
    sel = select_refined(cand, pad, l1, l2)[
        ["name", "noise", "seed", "score1", "q1", "area_ratio",
         "dice_refined", "iou_refined", "dice_refined_pp", "iou_refined_pp"]]
    f = base.merge(sel, on=["name", "noise", "seed"], how="inner")
    f["split"] = np.where(f.pid.map(lambda p: stable_seed(p) % 100 < VAL_FRACTION), "val", "test")
    return f


def methods(f, refine, metric, pp):
    s = "_pp" if pp else ""
    v = f[f"{metric}_vanilla{s}"].values
    r = f[f"{metric}_refined{s}"].values
    return {"vanilla": v, "ungated": r,
            "predicted_iou_gate": np.where(f.score1.values > f.score0.values, r, v),
            "ours": np.where(refine, r, v), "oracle": np.maximum(v, r),
            "medsam": f[f"{metric}_medsam{s}"].values, "search": f[f"{metric}_search{s}"].values}


def gate(name, f, p):
    s = (f.score1 - f.score0).values
    dq = (f.q1 - f.q0).values
    ar = f.area_ratio.values
    base = (f.score1 > f.score0).values
    if name == "K":
        return base | ((ar >= p["R"]) & (dq >= p["tau_d"]) & (s >= p["tau_s"]))
    if name == "I":
        ov = (ar >= p["R"]) & (dq >= p["tau_d"]) & (s >= p["tau_s"])
        veto = (s <= p["tau_bad"]) & (ar <= p["R_bad"])
        return (base | ov) & ~veto
    if name == "J":
        lc = np.log(np.clip(ar, 1e-6, None))
        hurt = ((s < p["tau_bad"]) & (ar < p["R_bad"])).astype(float)
        return (p["a"] * s + p["b"] * dq + p["c"] * lc - p["d"] * hurt) > p["thr"]
    raise ValueError(name)


GRIDS = {"I": GRID_I, "J": GRID_J, "K": GRID_K}


def val_objective(fval, refine, pp):
    d = methods(fval, refine, "dice", pp)
    i = methods(fval, refine, "iou", pp)
    d0 = (fval.noise == 0).values
    dreg = float((d["vanilla"][d0] - d["ours"][d0]).mean()) if d0.any() else 0.0
    ireg = float((i["vanilla"][d0] - i["ours"][d0]).mean()) if d0.any() else 0.0
    obj = float(((d["ours"] + i["ours"]) / 2).mean())
    return obj, (dreg <= REG_BUDGET and ireg <= REG_BUDGET), dreg, ireg


def sweep_gate(base, cand, gate_name, selections, pp):
    best = None
    for pad, l1, l2 in selections:
        f = build_frame(base, cand, pad, l1, l2)
        fval = f[f.split == "val"]
        for params in GRIDS[gate_name]:
            obj, feas, dreg, ireg = val_objective(fval, gate(gate_name, fval, params), pp)
            if feas and (best is None or obj > best["val_obj"]):
                best = dict(gate=gate_name, pad=pad, l1=l1, l2=l2, params=params,
                            val_obj=obj, val_dreg=dreg, val_ireg=ireg)
    return best


# ── test-side metrics + tables ───────────────────────────────────────────
def test_metrics(base, cand, cfg_best, pp):
    f = build_frame(base, cand, cfg_best["pad"], cfg_best["l1"], cfg_best["l2"])
    test = f[f.split == "test"].reset_index(drop=True)
    refine = gate(cfg_best["gate"], test, cfg_best["params"])
    d = methods(test, refine, "dice", pp)
    i = methods(test, refine, "iou", pp)
    d0 = (test.noise == 0).values
    hi = test.noise.isin([20, 30]).values
    orf = (test.dice_refined > test.dice_vanilla).values
    m = dict(
        overall_dice=float(d["ours"].mean()), overall_iou=float(i["ours"].mean()),
        d0_regression_dice=float((d["vanilla"][d0] - d["ours"][d0]).mean()),
        d0_regression_iou=float((i["vanilla"][d0] - i["ours"][d0]).mean()),
        hi_gap_vs_ungated=float((d["ours"][hi] - d["ungated"][hi]).mean()),
        loss_to_oracle=float((d["oracle"] - d["ours"]).mean()),
        gain_vs_predIoU=float((d["ours"] - d["predicted_iou_gate"]).mean()),
        refine_rate=float(refine.mean()),
        oracle_agreement=float((refine == orf).mean()))
    return test, refine, d, i, m


def table(test, d_or_i):
    rows = []
    for (ds, n), idx in test.groupby(["dataset", "noise"]).groups.items():
        pos = test.index.get_indexer(idx)
        row = {"dataset": ds, "noise": n}
        for meth in TABLE_METHODS:
            row[meth] = round(float(np.nanmean(d_or_i[meth][pos])), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def per_dataset(test, refine, d, i):
    orf = (test.dice_refined > test.dice_vanilla).values
    rows = []
    for ds in dict.fromkeys(test.dataset):
        m = (test.dataset == ds).values
        d0 = m & (test.noise == 0).values
        hi = m & test.noise.isin([20, 30]).values
        rows.append(dict(dataset=ds, n=int(m.sum()),
                         dice=d["ours"][m].mean(), iou=i["ours"][m].mean(),
                         d0_reg_dice=(d["vanilla"][d0] - d["ours"][d0]).mean() if d0.any() else np.nan,
                         hi_gain_vs_vanilla=(d["ours"][hi] - d["vanilla"][hi]).mean() if hi.any() else np.nan,
                         hi_gap_vs_ungated=(d["ours"][hi] - d["ungated"][hi]).mean() if hi.any() else np.nan,
                         loss_to_oracle=(d["oracle"][m] - d["ours"][m]).mean(),
                         refine_rate=refine[m].mean(),
                         oracle_agreement=(refine[m] == orf[m]).mean()))
    return pd.DataFrame(rows)


def promise_diag(test, refine, d, i):
    m = (test.dataset == "PROMISE12").values
    if m.sum() == 0:
        return None
    orf = (test.dice_refined > test.dice_vanilla).values
    noise = test.noise.values
    d0, hi = m & (noise == 0), m & np.isin(noise, [20, 30])
    return dict(
        clean_correctly_vetoed=int((d0 & ~refine & ~orf).sum()),
        clean_wrongly_refined=int((d0 & refine & ~orf).sum()),
        noisy_correctly_refined=int((hi & refine & orf).sum()),
        noisy_wrongly_vetoed=int((hi & ~refine & orf).sum()),
        hi_dice_vs_vanilla=float((d["ours"][hi] - d["vanilla"][hi]).mean()) if hi.any() else np.nan,
        hi_dice_vs_ungated=float((d["ours"][hi] - d["ungated"][hi]).mean()) if hi.any() else np.nan,
        hi_dice_vs_gate=float((d["ours"][hi] - d["predicted_iou_gate"][hi]).mean()) if hi.any() else np.nan,
        hi_iou_vs_vanilla=float((i["ours"][hi] - i["vanilla"][hi]).mean()) if hi.any() else np.nan,
        hi_iou_vs_ungated=float((i["ours"][hi] - i["ungated"][hi]).mean()) if hi.any() else np.nan)


def _heatmap(M, rows, cols, title, fname, out, cmap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2.5, 0.7 * len(rows) + 2))
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"δ={c}" for c in cols])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    for r in range(len(rows)):
        for c in range(len(cols)):
            if not np.isnan(M[r, c]):
                ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title(title); fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(Path(out) / fname, dpi=150, bbox_inches="tight"); plt.close(fig)


def _grid(test, values_fn):
    dsets = list(dict.fromkeys(test.dataset)); noises = sorted(test.noise.unique())
    M = np.full((len(dsets), len(noises)), np.nan)
    for r, ds in enumerate(dsets):
        for c, n in enumerate(noises):
            mask = ((test.dataset == ds) & (test.noise == n)).values
            if mask.any():
                M[r, c] = values_fn(mask)
    return M, dsets, noises


def figures(test, refine, d, i, summ, out):
    for metric, dd, name in [("dice", d, "fig_final_dice_heatmap.png"),
                             ("iou", i, "fig_final_iou_heatmap.png")]:
        M, rs, cs = _grid(test, lambda mk, dd=dd: dd["ours"][mk].mean())
        _heatmap(M, rs, cs, f"Final {metric.upper()} (ours)", name, out, "viridis")
    M, rs, cs = _grid(test, lambda mk: (d["oracle"][mk] - d["ours"][mk]).mean())
    _heatmap(M, rs, cs, "Loss to oracle (Dice)", "fig_loss_to_oracle.png", out, "magma")
    M, rs, cs = _grid(test, lambda mk: refine[mk].mean())
    _heatmap(M, rs, cs, "Refine rate", "fig_refine_rate.png", out, "viridis")
    orf = (test.dice_refined > test.dice_vanilla).values
    M, rs, cs = _grid(test, lambda mk: (refine[mk] == orf[mk]).mean())
    _heatmap(M, rs, cs, "Oracle agreement", "fig_oracle_agreement.png", out, "cividis")
    # pareto over the summary gate rows
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, r in summ.iterrows():
        ax.scatter(r.d0_regression_dice, r.loss_to_oracle, s=60)
        ax.annotate(r.method, (r.d0_regression_dice, r.loss_to_oracle), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axvline(REG_BUDGET, color="r", ls="--", lw=1, label=f"δ0 budget {REG_BUDGET}")
    ax.set_xlabel("δ=0 Dice regression vs vanilla"); ax.set_ylabel("loss to oracle (Dice)")
    ax.set_title("Pareto: clean-prompt safety vs oracle gap"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(Path(out) / "fig_pareto.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ── stages ───────────────────────────────────────────────────────────────
def _paths(cfg):
    out = _common.out_dir(cfg, "clean_veto_opt")
    return out, out / "base.csv", out / "cand.csv"


def run_cache(cfg, args):
    out, base_p, cand_p = _paths(cfg)
    sam = _common.build_sam(cfg)
    try:
        medsam = _common.build_medsam(cfg)
    except Exception as exc:
        print(f"[warn] MedSAM unavailable ({exc})"); medsam = None
    base, cand = [], []
    for name in _common.dataset_names(cfg, args):
        samples = load_dataset(name, cfg)
        if samples:
            b, c = cache(sam, medsam, samples, cfg, name, do_search=not args.skip_search)
            base += b; cand += c
    pd.DataFrame(base).to_csv(base_p, index=False)
    pd.DataFrame(cand).to_csv(cand_p, index=False)
    print(f"\ncached {len(base)} cells, {len(cand)} candidates -> {out}")


def run_analyze(cfg, args):
    out, base_p, cand_p = _paths(cfg)
    base = pd.read_csv(base_p)
    cand = pd.read_csv(cand_p)
    pp = bool(args.postproc)
    selections = [(pad, l1, l2) for pad in PAD_GRID for l1 in LAM1 for l2 in LAM2]

    best = {g: sweep_gate(base, cand, g, selections, pp) for g in ["I", "J", "K"]}
    best_I_old = sweep_gate(base, cand, "I", [(0.0, 0.0, 0.0)], pp)   # adopted gate I (reference candidate)
    final_name = max(["I", "J", "K"], key=lambda g: (best[g] or {"val_obj": -1})["val_obj"])
    final_cfg = best[final_name]
    if final_cfg is None:
        print("No feasible config under the δ=0 budget; keeping gate I.")
        final_cfg, final_name = best_I_old, "I"

    test, refine, d, i, mfin = test_metrics(base, cand, final_cfg, pp)

    # gate-summary rows (each best-on-val gate evaluated on test) + references
    rows = []
    refs = {"vanilla": np.zeros(len(test), bool), "ungated": np.ones(len(test), bool),
            "predicted_iou_gate": (test.score1 > test.score0).values}
    for nm, rmask in refs.items():
        dd = methods(test, rmask, "dice", pp); ii = methods(test, rmask, "iou", pp)
        d0 = (test.noise == 0).values; hi = test.noise.isin([20, 30]).values
        orf = (test.dice_refined > test.dice_vanilla).values
        rows.append(dict(method=nm, overall_dice=dd["ours"].mean(), overall_iou=ii["ours"].mean(),
                         d0_regression_dice=(dd["vanilla"][d0] - dd["ours"][d0]).mean(),
                         d0_regression_iou=(ii["vanilla"][d0] - ii["ours"][d0]).mean(),
                         hi_gap_vs_ungated=(dd["ours"][hi] - dd["ungated"][hi]).mean(),
                         loss_to_oracle=(dd["oracle"] - dd["ours"]).mean(),
                         gain_vs_predIoU=(dd["ours"] - dd["predicted_iou_gate"]).mean(),
                         refine_rate=rmask.mean(), oracle_agreement=(rmask == orf).mean(), feasible=True))
    for label, cfgb in [("gate_I_old", best_I_old), ("gate_J", best["J"]),
                        ("gate_K", best["K"]), ("final_ours", final_cfg)]:
        if cfgb is None:
            continue
        _, _, _, _, mm = test_metrics(base, cand, cfgb, pp)
        rows.append(dict(method=label, feasible=True, **{k: mm[k] for k in mm}))
    # oracle row
    dd = methods(test, np.ones(len(test), bool), "dice", pp); ii = methods(test, np.ones(len(test), bool), "iou", pp)
    rows.append(dict(method="oracle", overall_dice=dd["oracle"].mean(), overall_iou=ii["oracle"].mean(),
                     d0_regression_dice=0.0, d0_regression_iou=0.0, hi_gap_vs_ungated=np.nan,
                     loss_to_oracle=0.0, gain_vs_predIoU=np.nan, refine_rate=np.nan,
                     oracle_agreement=1.0, feasible=True))
    summ = pd.DataFrame(rows)

    # write everything
    table(test, d).to_csv(out / "final_table_dice.csv", index=False)
    table(test, i).to_csv(out / "final_table_iou.csv", index=False)
    summ.to_csv(out / "gate_summary.csv", index=False)
    pds = per_dataset(test, refine, d, i); pds.to_csv(out / "per_dataset_summary.csv", index=False)
    pdiag = promise_diag(test, refine, d, i)
    if pdiag is not None:
        pd.DataFrame([pdiag]).to_csv(out / "PROMISE12_diagnosis.csv", index=False)
    figures(test, refine, d, i, summ, out)
    (out / "final_config.json").write_text(json.dumps({"gate": final_name, **{k: final_cfg[k]
                                          for k in ("pad", "l1", "l2", "params")}}, default=str))

    # print
    pd.set_option("display.width", 220)
    print(f"\nVAL/TEST {VAL_FRACTION}/{100-VAL_FRACTION} by patient; config selected on VAL only "
          f"(max mean(Dice+IoU)/2 s.t. δ0 Dice & IoU regression ≤ {REG_BUDGET}); postproc={pp}.")
    print(f"\nFINAL config: gate {final_name}  pad={final_cfg['pad']} λ1={final_cfg['l1']} "
          f"λ2={final_cfg['l2']}  params={final_cfg['params']}")
    print("\n=== FINAL Table 1 — Dice (TEST) ===");  print(table(test, d).to_string(index=False))
    print("\n=== FINAL Table 1 — IoU (TEST) ===");   print(table(test, i).to_string(index=False))
    print("\n=== GATE SUMMARY (TEST) ===")
    print(summ.round(4).to_string(index=False))
    print("\n=== PER-DATASET (TEST) ==="); print(pds.round(3).to_string(index=False))
    if pdiag is not None:
        print("\n=== PROMISE12 diagnosis (TEST) ===")
        print("  " + "  ".join(f"{k}={v}" for k, v in pdiag.items()))

    gate_dice = float(summ.loc[summ.method == "predicted_iou_gate", "overall_dice"].iloc[0])
    gate_iou = float(summ.loc[summ.method == "predicted_iou_gate", "overall_iou"].iloc[0])
    old = best_I_old and test_metrics(base, cand, best_I_old, pp)[4]
    old_dice = old["overall_dice"] if old else float("nan")
    safe = (mfin["overall_dice"] >= gate_dice - 1e-9 and mfin["overall_dice"] >= old_dice - 1e-9
            and mfin["d0_regression_dice"] <= REG_BUDGET and mfin["d0_regression_iou"] <= REG_BUDGET)
    print("\n" + "=" * 78)
    print("FINAL VERDICT")
    print(f"  gate I (old)      : Dice {old_dice:.3f}")
    print(f"  predicted-IoU gate: Dice {gate_dice:.3f}  IoU {gate_iou:.3f}")
    print(f"  FINAL ours ({final_name}) : Dice {mfin['overall_dice']:.3f}  IoU {mfin['overall_iou']:.3f}")
    print(f"  gain vs predicted-IoU gate (Dice): {mfin['gain_vs_predIoU']:+.3f}")
    print(f"  loss-to-oracle (Dice): {mfin['loss_to_oracle']:.3f}")
    print(f"  δ=0 regression: Dice {mfin['d0_regression_dice']:+.3f}  IoU {mfin['d0_regression_iou']:+.3f}")
    if safe:
        print("  -> SAFE TO ADOPT: final ours improves on gate I and the predicted-IoU gate within budget.")
    else:
        print("  -> NOT AN IMPROVEMENT within budget. KEEP gate I; do not tune further.")
    print("=" * 78)
    print(f"\nAll tables + figures written to {out}")


def main(argv=None):
    p = _common.base_parser(__doc__)
    p.add_argument("--stage", choices=["cache", "analyze"], required=True)
    p.add_argument("--postproc", action="store_true", help="apply CC+hole-fill to ALL methods")
    p.add_argument("--skip-search", action="store_true")
    args = p.parse_args(argv)
    cfg = _common.get_config(args)
    if args.stage == "cache":
        run_cache(cfg, args)
    else:
        run_analyze(cfg, args)


if __name__ == "__main__":
    main()

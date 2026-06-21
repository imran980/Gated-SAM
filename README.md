# Gated-SAM → Consistency-Driven Prompt-Space Optimization

Reproducible code for the **WACV 2027** reframe of *Gated Recursive Refinement*. The
method makes SAM robust to noisy box prompts at inference time, with **no training and
no auxiliary models**, by:

1. a **reference-free objective** `Q` — *perturbation-consistency*: the mean pairwise
   IoU of masks produced from `K` jittered versions of a box prompt. High `Q` means the
   mask is a fixed point of the prompt→mask map, which empirically tracks true Dice
   better than SAM's predicted-IoU head under domain shift;
2. a **prompt-space search** that maximizes `Q` over a candidate neighborhood
   (tight box, dilate±, erode±, jittered boxes, the 3 multimask outputs); and
3. a **guarded return** — the best-`Q` mask over the *whole* trajectory, including the
   original single-pass prediction. This gives a **no-regression guarantee** and fixes
   the δ=0 lock-in for free (a clean prompt is already jitter-stable, so the search
   keeps it).

> Every number in the paper comes from a script in `src/gated_sam/experiments/`. There
> are **no hardcoded result tables** (see `docs/reframe.md` for why this matters).

## Install

```bash
pip install -e .            # the gated_sam package + console scripts
pip install -r requirements.txt   # torch, segment-anything, SimpleITK, ...
pytest -q                   # 18 GPU-free tests (mock predictor) — run this first
```

## Data & checkpoints layout

Paths are set in `configs/default.yaml` (relative to `data_root` / `checkpoint_root`):

```
data/
  jsrt/jpg/*.jpg            jsrt/masks/*.tif
  Dataset_BUSI_with_GT/{benign,malignant}/*.png  (+ *_mask.png)
  Kvasir-SEG/images/*.jpg   Kvasir-SEG/masks/*.jpg
  promise12/{train_data,test_data}/*.mhd  (+ *_segmentation.mhd)
checkpoints/
  sam_vit_b_01ec64.pth      medsam_vit_b.pth
```

Override anything on the CLI: `--set device=cuda n_images_per_dataset=100 search.max_steps=3`.

## The 7-day workflow → commands

| Day | Goal | Command |
|-----|------|---------|
| 1 | **Go/no-go**: do the signals correlate with true Dice? (Figure 2) | `python -m gated_sam.experiments.day1_correlation` |
| 2–3 | Build/verify the optimizer (covered by `pytest`; logic in `refine.py`) | `pytest -q` |
| 4–5 | **Main table**: Dice+HD95, δ∈{0,10,20,30}, seeds, CIs, Wilcoxon | `python -m gated_sam.experiments.main_table` |
| 6 | Lock-in / fixed-point analysis + recovery rate | `python -m gated_sam.experiments.lockin --noise 30` |
| 6 | Ablations: objective, candidate set, #steps, guard | `python -m gated_sam.experiments.ablations --noise 30` |

Day 1 prints a **GO/NO-GO verdict**: proceed only if perturbation-consistency beats
predicted-IoU, especially on BUSI/PROMISE12. Day 5 prints a **contribution check**:
`Ours > predicted-IoU gate > ungated`, with paired-Wilcoxon significance.

Quick smoke run (tiny, to validate the pipeline on your box before a full sweep):

```bash
python -m gated_sam.experiments.day1_correlation --set n_images_per_dataset=5 seeds=[0,1]
```

## Package map

```
src/gated_sam/
  metrics.py      Dice, IoU, HD95, mask→box
  prompts.py      box noise / jitter / dilate / erode / candidate neighborhood
  objectives.py   predicted_iou | coarse_agreement | perturbation_consistency | combo
  refine.py       refinement map T + guarded consistency search (+ trajectory logging)
  baselines.py    vanilla SAM, MedSAM, ungated cascade, predicted-IoU gate, Ours
  models.py       SAM/MedSAM predictor wrapper + MockPredictor (GPU-free tests)
  data.py         JSRT / BUSI / Kvasir / PROMISE12 loaders → unified Sample
  stats.py        mean±95% CI, bootstrap, paired Wilcoxon
  figures.py      Figure 2, robustness curves, lock-in trajectories
  experiments/    day1_correlation · main_table · lockin · ablations
```

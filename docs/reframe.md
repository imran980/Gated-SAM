# MICCAI → WACV reframe: what changed and why

## Why MICCAI rejected it
- **Low novelty.** "Two-pass refine + confidence gate" reads as iterative box refinement
  (PerSAM-style) plus a predicted-IoU gate — both known. The contribution wasn't isolated.
- **Weak results comparison.** The main table compared only SAM / MedSAM / Ours, and the
  reproducible numbers in the repo did **not** match the paper.

## Two integrity problems in the submitted version (must not recur)
1. **Hardcoded / inconsistent headline numbers.** Notebook cell 51 is literally
   `# === HARDCODED RESULTS ===`. It (and the README, and ablation Table 3) report
   **JSRT Ours @ δ=30 = 0.906**, but the abstract, Table 1, and Contribution 4 say
   **0.977**. SAM @ δ=30 is 0.848 vs 0.854. The main table appears hand-edited upward.
   → In this repo, **all** tables are emitted by scripts. Nothing is typed by hand.
2. **δ=0 regression.** At clean prompts the old method is *worse* than SAM on every
   dataset (PROMISE12: 0.761 vs 0.911). The guarded return removes this by construction.

## The reframe (new contributions)
1. **Reference-free objective.** Perturbation-consistency `Q` — needs no ground truth and
   no CLIP/DINO. Day-1 proves it correlates with true Dice better than predicted-IoU
   (Figure 2). This makes the objective the search optimizes *empirically real*, not asserted.
2. **Prompt-space optimization.** Treat the prompt as the optimization variable; search a
   candidate neighborhood for `argmax Q` instead of doing a single blind refinement pass.
3. **Guarded no-regression property.** Return the best-`Q` mask over the whole trajectory
   (incl. the original). Guarantees no regression on clean prompts; fixes δ=0 for free.
4. **Fixed-point / lock-in characterization.** Show lock-in is the flat-/low-`Q` regime,
   quantify the recovery rate (headline: PROMISE12), and show the search escapes where
   plain iteration cannot.

## The comparison that proves the contribution
One ordering, measured with CIs + paired Wilcoxon, at δ∈{0,10,20,30}:

```
Ours (consistency search)  >  predicted-IoU gate (old)  >  ungated cascade (PerSAM)
```

- `ungated_cascade` isolates "refinement helps at all".
- `predicted_iou_gate` isolates "gating on the OLD signal".
- `ours` isolates "gating/searching on the consistency objective".

If `ours > gate` is significant where δ≥20, the new objective — not just the refinement —
is what carries the gain. That is the novelty, measured rather than claimed.

## Method note vs. the old paper
- Old "δ=0 bug": clean `M1` is jitter-stable → high `Q` → the guard keeps it. No tuning.
- "Lock-in": old paper described it as a limitation; here it becomes an analyzed
  phenomenon (trajectory `Q` and step-movement plots) and a quantified recovery rate.
- Backbone: paper states ViT-B; config defaults to ViT-B for SAM **and** MedSAM so the
  comparison is backbone-matched (the old notebook config used vit_h — a confound).

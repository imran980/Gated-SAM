"""Figure generation. Matplotlib only; safe to import without a display."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SIGNAL_COLORS = {
    "predicted_iou": "#E74C3C",          # old gate (weak)
    "coarse_agreement": "#F39C12",
    "perturbation_consistency": "#3498DB",  # ours
}
SIGNAL_LABELS = {
    "predicted_iou": "Predicted-IoU (old gate)",
    "coarse_agreement": "Coarse agreement",
    "perturbation_consistency": "Consistency (ours)",
}


def _save(fig, out_dir: Path, name: str):
    out_dir = Path(out_dir)
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def day1_correlation_figure(df, sp, out_dir: Path):
    """Figure 2: (a) Spearman rho of each signal per dataset; (b) consistency-vs-Dice scatter."""
    datasets = [d for d in sp["dataset"].tolist() if d != "POOLED"]
    signals = list(SIGNAL_COLORS)

    fig = plt.figure(figsize=(6 + 2.4 * len(datasets), 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, max(1.0, 0.75 * len(datasets))])

    # (a) grouped bar chart of Spearman rho
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(datasets))
    width = 0.26
    for i, sig in enumerate(signals):
        vals = [float(sp.loc[sp["dataset"] == d, f"{sig}_rho"].iloc[0]) for d in datasets]
        ax.bar(x + (i - 1) * width, vals, width, color=SIGNAL_COLORS[sig], label=SIGNAL_LABELS[sig])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(r"Spearman $\rho$ vs. true Dice")
    ax.set_title("(a) Signal–quality correlation")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    # (b) scatter of the winning signal vs true Dice, pooled, colored by dataset
    ax2 = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("tab10")
    for j, d in enumerate(datasets):
        g = df[df["dataset"] == d]
        ax2.scatter(g["perturbation_consistency"], g["true_dice"], s=10, alpha=0.5,
                    color=cmap(j % 10), label=d)
    ax2.set_xlabel("Perturbation-consistency (reference-free)")
    ax2.set_ylabel("True Dice")
    ax2.set_title("(b) Consistency tracks true Dice")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "figure2_correlation")


def robustness_curves(table_df, out_dir: Path, methods, method_styles=None):
    """Dice-vs-noise curves, one subplot per dataset, mean with CI band if present."""
    datasets = list(dict.fromkeys(table_df["dataset"]))
    styles = method_styles or {}
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), squeeze=False)
    for idx, d in enumerate(datasets):
        ax = axes[0][idx]
        sub = table_df[table_df["dataset"] == d].sort_values("noise")
        for m in methods:
            col = f"{m}_dice_mean"
            if col not in sub:
                continue
            st = styles.get(m, {})
            ax.plot(sub["noise"], sub[col], marker=st.get("marker", "o"),
                    color=st.get("color"), label=st.get("label", m), lw=2)
            half = f"{m}_dice_ci"
            if half in sub:
                ax.fill_between(sub["noise"], sub[col] - sub[half], sub[col] + sub[half],
                                color=st.get("color"), alpha=0.15)
        ax.set_title(d, fontweight="bold")
        ax.set_xlabel("Box noise (px)")
        ax.set_ylabel("Dice")
        ax.set_xticks(sorted(sub["noise"].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    _save(fig, out_dir, "figure_robustness_curves")


def lockin_trajectories(traces, out_dir: Path):
    """Q(M_k) and step-movement IoU(M_k, M_{k-1}) for recovery vs lock-in cases."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for tr in traces:
        ks = [s["k"] for s in tr["steps"]]
        axes[0].plot(ks, [s["Q"] for s in tr["steps"]], marker="o", alpha=0.7,
                     color="#2ecc71" if tr["recovered"] else "#e74c3c")
        axes[1].plot(ks, [s["move"] for s in tr["steps"]], marker="s", alpha=0.7,
                     color="#2ecc71" if tr["recovered"] else "#e74c3c")
    axes[0].set_title("Objective Q along the trajectory")
    axes[0].set_xlabel("step k"); axes[0].set_ylabel("Q(M_k)")
    axes[1].set_title("Step movement IoU(M_k, M_{k-1})")
    axes[1].set_xlabel("step k"); axes[1].set_ylabel("movement")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Green = recovery, Red = lock-in", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "figure_lockin_trajectories")

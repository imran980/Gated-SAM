"""Aggregation statistics for the main table: mean ± 95% CI, paired Wilcoxon."""
from __future__ import annotations

import numpy as np


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a normal-approx CI. Returns (mean, half_width)."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    # 1.96 for 95%; good enough for n>=~20, conservative otherwise.
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    se = arr.std(ddof=1) / np.sqrt(arr.size)
    return mean, float(z * se)


def bootstrap_ci(values, confidence: float = 0.95, n_boot: int = 2000,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (mean, lo, hi)."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def wilcoxon(a, b):
    """Paired Wilcoxon signed-rank test (ours vs baseline), per-image paired values.

    Returns (statistic, p_value). Pairs with zero difference are dropped. Returns
    (nan, 1.0) when there is no signal to test.
    """
    from scipy.stats import wilcoxon as _w

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size == 0 or np.allclose(a, b):
        return float("nan"), 1.0
    try:
        stat, p = _w(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return float("nan"), 1.0


def fmt_ci(mean: float, half: float, decimals: int = 3) -> str:
    if np.isnan(mean):
        return "—"
    return f"{mean:.{decimals}f} ± {half:.{decimals}f}"

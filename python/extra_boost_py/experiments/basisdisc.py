"""Discovered extrapolation basis: periodogram -> Fourier + trend (train window only).

Real datasets carry no ready-made extrapolating features; the leaf basis must be
constructed. This module finds dominant periodicities of the target on the TRAIN
window (Lomb-Scargle periodogram of the binned aggregate curve, detrended) and turns
them into sin/cos basis columns for the Bench.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.signal import lombscargle

from .benchgen import Bench


@dataclass(frozen=True)
class DiscoveredBasis:
    freqs: Tuple[float, ...]      # cycles per unit of normalized t
    r2_train: float               # basis R^2 on the aggregate train curve


def _aggregate_curve(bench: Bench, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    tr = bench.train
    t = tr["t"].to_numpy()
    y = tr["y"].to_numpy()
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=y, minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    mask = counts > 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[mask], sums[mask] / counts[mask]


def _detrend(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def _basis_matrix(t: np.ndarray, freqs: Tuple[float, ...]) -> np.ndarray:
    cols = [np.ones_like(t), t]
    for f in freqs:
        cols.append(np.sin(2.0 * np.pi * f * t))
        cols.append(np.cos(2.0 * np.pi * f * t))
    return np.column_stack(cols)


def discover_basis(bench: Bench, max_freqs: int = 3, n_bins: int = 1600,
                   min_power_ratio: float = 4.0) -> DiscoveredBasis:
    tb, yb = _aggregate_curve(bench, n_bins)
    if len(tb) < 16:
        return DiscoveredBasis(freqs=(), r2_train=0.0)
    resid = _detrend(tb, yb)
    span = tb.max() - tb.min()
    f_lo, f_hi = 2.0 / span, n_bins / (4.0 * span)
    f_grid = np.linspace(f_lo, f_hi, 4000)
    power = lombscargle(tb, resid, 2.0 * np.pi * f_grid, normalize=False)

    # significance floor: max periodogram power of permuted residuals (destroys any
    # temporal structure while keeping the value distribution), with safety margin
    rng = np.random.default_rng(0)
    perm_max = max(
        lombscargle(tb, rng.permutation(resid), 2.0 * np.pi * f_grid,
                    normalize=False).max()
        for _ in range(3)
    )
    floor = max(1.1 * perm_max, min_power_ratio * float(np.median(power)))
    freqs = []
    p = power.copy()
    for _ in range(max_freqs):
        i = int(np.argmax(p))
        if p[i] <= floor:
            break
        f_peak = float(f_grid[i])
        freqs.append(f_peak)
        # suppress the whole broad peak: proportional exclusion zone (+-15%)
        zone = (f_grid > 0.85 * f_peak) & (f_grid < 1.15 * f_peak)
        p[zone] = 0.0

    freqs_t = tuple(sorted(freqs))
    A = _basis_matrix(tb, freqs_t)
    coef, *_ = np.linalg.lstsq(A, yb, rcond=None)
    ss_res = float(np.sum((yb - A @ coef) ** 2))
    ss_tot = float(np.sum((yb - yb.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DiscoveredBasis(freqs=freqs_t, r2_train=r2)


def apply_basis(bench: Bench, basis: DiscoveredBasis) -> Bench:
    """New Bench with sin/cos columns appended to BOTH extra_cols (GBDTE leaf basis)
    and partition_cols (fairness: external models see them as ordinary features)."""
    df = bench.df.copy()
    extra_cols = list(bench.extra_cols)
    partition_cols = list(bench.partition_cols)
    t = df["t"].to_numpy()
    next_e = len(extra_cols)
    for f in basis.freqs:
        for func in (np.sin, np.cos):
            col = f"e_{next_e}"
            df[col] = func(2.0 * np.pi * f * t)
            extra_cols.append(col)
            partition_cols.append(col)
            next_e += 1
    return Bench(df=df, partition_cols=partition_cols, extra_cols=extra_cols,
                 task=bench.task, cut=bench.cut, config=bench.config)


__all__ = ["DiscoveredBasis", "discover_basis", "apply_basis"]

"""E4/E5: data-computable role-separability statistics and automatic role assignment."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .benchgen import Bench


def drift_score(x: np.ndarray, t: np.ndarray) -> float:
    """How much a single feature carries time signal: folded AUC of x vs late-half."""
    late = (t > np.median(t)).astype(int)
    if np.unique(x).size < 2:
        return 0.5
    auc = roc_auc_score(late, x)
    return float(max(auc, 1.0 - auc))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def stability_score(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """Cross-time stability of the feature-target relation: min |corr| over time halves
    if the sign agrees, else 0."""
    med = np.median(t)
    early, late = t <= med, t > med
    c1, c2 = _corr(x[early], y[early]), _corr(x[late], y[late])
    if c1 * c2 <= 0.0:
        return 0.0
    return float(min(abs(c1), abs(c2)))


def separability_index(bench: Bench) -> float:
    """Mean stability of binary partition candidates; tracks role purity (rho)."""
    df = bench.df
    y, t = df["y"].to_numpy(), df["t"].to_numpy()
    scores = []
    for col in bench.partition_cols:
        x = df[col].to_numpy()
        if np.unique(x).size <= 2:  # binary candidates only
            scores.append(stability_score(x, y, t))
    return float(np.mean(scores)) if scores else 0.0


def auto_roles(bench: Bench, drift_threshold: float = 0.6,
               max_leaf: int = 4) -> Tuple[List[str], List[str]]:
    """Assign candidate columns to partition vs leaf blocks by drift score.

    At most `max_leaf` drifting features join the leaf block (highest drift first);
    near-constant columns never do — both guards keep the per-leaf linear system
    well-conditioned on real data."""
    df = bench.df
    t = df["t"].to_numpy()
    partition, drifting = [], []
    for col in bench.partition_cols:
        x = df[col].to_numpy()
        score = drift_score(x, t)
        if score >= drift_threshold and np.std(x) > 1e-12:
            drifting.append((score, col))
        else:
            partition.append(col)
    drifting.sort(reverse=True)
    leaf = ["t"] + [col for _, col in drifting[:max_leaf]]
    partition += [col for _, col in drifting[max_leaf:]]
    return partition, leaf


__all__ = ["drift_score", "stability_score", "separability_index", "auto_roles"]

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


def auto_roles(bench: Bench, drift_threshold: float = 0.6) -> Tuple[List[str], List[str]]:
    """Assign candidate columns to partition vs leaf blocks by drift score."""
    df = bench.df
    t = df["t"].to_numpy()
    partition, leaf = [], ["t"]
    for col in bench.partition_cols:
        if drift_score(df[col].to_numpy(), t) >= drift_threshold:
            leaf.append(col)
        else:
            partition.append(col)
    return partition, leaf


__all__ = ["drift_score", "stability_score", "separability_index", "auto_roles"]

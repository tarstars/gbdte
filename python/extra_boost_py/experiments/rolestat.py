"""E4/E5: data-computable role-separability statistics and automatic role assignment."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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


def _binned_stability(x: np.ndarray, y: np.ndarray, t: np.ndarray, n_bins: int = 10) -> float:
    """Stability of the x->y relation across time halves, for x of any cardinality.
    Bin x into <=n_bins quantile groups, compare per-bin mean-y between the early and
    late time halves via Spearman rank correlation (0 if too few bins)."""
    med = np.median(t)
    early, late = t <= med, t > med
    if early.sum() < 2 or late.sum() < 2:
        return 0.0
    uniq = np.unique(x)
    if uniq.size <= n_bins:
        bins = x
    else:
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.digitize(x, edges)
    ea = pd.Series(y[early]).groupby(pd.Series(bins[early])).mean()
    la = pd.Series(y[late]).groupby(pd.Series(bins[late])).mean()
    common = ea.index.intersection(la.index)
    if len(common) < 3:
        return 0.0
    rho, _ = spearmanr(ea.loc[common].to_numpy(), la.loc[common].to_numpy())
    return float(max(rho, 0.0)) if np.isfinite(rho) else 0.0


def separability_index(bench: Bench, max_card: int = 10) -> float:
    """Mean cross-time stability of the target's dependence on CATEGORICAL partition
    features (cardinality <= max_card). Generalizes the old binary-only version to
    multi-way categoricals (region, income, class id) while keeping the validated
    discrete-group meaning and scale. Genuinely continuous group structure is caught by
    extrapolation_gain instead (a tree forms the groups there)."""
    df = bench.df
    y, t = df["y"].to_numpy(), df["t"].to_numpy()
    scores = []
    for c in bench.partition_cols:
        x = df[c].to_numpy()
        card = np.unique(x).size
        if card == 2:
            scores.append(stability_score(x, y, t))    # correlation-based (validated)
        elif card <= max_card:
            scores.append(_binned_stability(x, y, t))  # Spearman of per-bin means
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


def extrapolation_gain(bench: Bench, max_depth: int = 4) -> float:
    """Cheap screen for leaf-linear forward headroom: discover groups with a shallow
    tree on partition features (no time), then measure how much a per-group linear-in-t
    model beats a per-group constant on an inner-FORWARD slice of the train window.
    Relative improvement in the task metric; ~0 means no dataset trick will help."""
    from sklearn.metrics import log_loss
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    tr = bench.train
    if len(tr) < 200:
        return 0.0
    X = tr[bench.partition_cols].to_numpy(dtype=np.float64)
    y = tr["y"].to_numpy(dtype=np.float64)
    t = tr["t"].to_numpy(dtype=np.float64)
    q = np.quantile(t, 0.75)
    inner, fwd = t < q, t >= q
    if inner.sum() < 50 or fwd.sum() < 20:
        return 0.0

    Tree = DecisionTreeClassifier if bench.task == "logloss" else DecisionTreeRegressor
    tree = Tree(max_depth=max_depth, min_samples_leaf=50, random_state=0)
    tree.fit(X[inner], y[inner])
    leaf_all = tree.apply(X)
    lf_inner, lf_fwd = leaf_all[inner], leaf_all[fwd]

    def _fit_predict(linear: bool) -> np.ndarray:
        preds = np.empty(int(fwd.sum()))
        yf, tf = y[inner], t[inner]
        te = t[fwd]
        for lf in np.unique(lf_fwd):
            sel_fit = lf_inner == lf
            sel_eval = lf_fwd == lf
            if sel_fit.sum() >= 10 and linear:
                A = np.column_stack([np.ones(int(sel_fit.sum())), tf[sel_fit]])
                coef, *_ = np.linalg.lstsq(A, yf[sel_fit], rcond=None)
                p = np.column_stack([np.ones(int(sel_eval.sum())), te[sel_eval]]) @ coef
            else:
                p = np.full(int(sel_eval.sum()),
                            yf[sel_fit].mean() if sel_fit.any() else yf.mean())
            preds[np.where(sel_eval)[0]] = p
        return preds

    const_p, lin_p = _fit_predict(False), _fit_predict(True)
    yv = y[fwd]
    if bench.task == "logloss":
        c = log_loss(yv, np.clip(1 / (1 + np.exp(-const_p)), 1e-6, 1 - 1e-6), labels=[0, 1])
        l = log_loss(yv, np.clip(1 / (1 + np.exp(-lin_p)), 1e-6, 1 - 1e-6), labels=[0, 1])
    else:
        c = float(np.mean((yv - const_p) ** 2))
        l = float(np.mean((yv - lin_p) ** 2))
    return float(max((c - l) / c, 0.0)) if c > 0 else 0.0


__all__ = ["drift_score", "stability_score", "separability_index",
           "extrapolation_gain", "auto_roles"]

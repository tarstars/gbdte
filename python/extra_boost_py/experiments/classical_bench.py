"""Adapter: the paper's classical LogLoss benchmark as a Bench, with exact Bayes oracle."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..classical_dataset import ClassicalFeatureSpec, generate_classical_dataset
from .benchgen import Bench

_ASCEND = ClassicalFeatureSpec(slope=0.75, intercept=0.5, beta=0.2)
_DESCEND = ClassicalFeatureSpec(slope=-0.75, intercept=1.25, beta=0.1)


def _specs() -> list[ClassicalFeatureSpec]:
    specs = [_ASCEND, _DESCEND]
    for idx in range(15):
        specs.append(_DESCEND if idx < 7 else _ASCEND)
    return specs


def _oracle_logodds(features: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """Bayes log-odds: features are conditionally independent given the label, with
    P(f=1|y=1)=beta and P(f=1|y=0)=gamma(t) as in generate_classical_dataset."""
    denom = 1.0 / alpha - 1.0
    score = np.full(t.shape, np.log(alpha / (1.0 - alpha)))
    for col, s in enumerate(_specs()):
        lift = s.slope * t + s.intercept
        gamma = ((1.0 / (alpha * lift) - 1.0) / denom) * s.beta
        f = features[:, col]
        score += f * np.log(s.beta / gamma) + (1.0 - f) * np.log((1.0 - s.beta) / (1.0 - gamma))
    return score


def classical_bench(n_rows: int = 10000, seed: int = 0, cut: float = 0.4,
                    alpha: float = 0.3) -> Bench:
    ds = generate_classical_dataset(n_rows, alpha=alpha, seed=seed)
    n_feat = ds.features_inter.shape[1]
    data = {f"f_{i}": ds.features_inter[:, i] for i in range(n_feat)}
    data["t"] = ds.time
    data["e_0"] = np.ones_like(ds.time)
    data["e_1"] = ds.time
    data["y"] = ds.target
    data["oracle"] = _oracle_logodds(ds.features_inter, ds.time, alpha)
    return Bench(
        df=pd.DataFrame(data),
        partition_cols=[f"f_{i}" for i in range(n_feat)],
        extra_cols=["e_0", "e_1"],
        task="logloss",
        cut=cut,
    )


__all__ = ["classical_bench"]

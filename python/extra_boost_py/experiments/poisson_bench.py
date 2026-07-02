"""Poisson benchmark: grouped drifting event intensity with role separation.

Each object is a cohort of `exposure` independent units observed over [0, 1] in
`n_bins` time bins; per-bin counts are Poisson with the cohort intensity. Counts are
clipped to >= 1 because the legacy engine cannot take zero frequencies (see
test_poisson_semantics.py for the validated engine semantics).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PoissonBenchConfig:
    k_groups: int = 8
    rho: float = 1.0
    drift: float = 1.0
    n_objects: int = 500
    n_bins: int = 20
    exposure: int = 20          # cohort size: per-bin counts ~ Pois(lam * delta * exposure)
    cut: float = 0.6
    seed: int = 0


@dataclass
class PoissonBench:
    df: pd.DataFrame
    partition_cols: List[str]
    extra_cols: List[str]
    cut: float
    delta: float
    task: str = "poisson"
    config: Optional[PoissonBenchConfig] = None

    @property
    def train(self) -> pd.DataFrame:
        return self.df[self.df["t"] < self.cut]

    @property
    def test(self) -> pd.DataFrame:
        return self.df[self.df["t"] >= self.cut]

    def event_train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tr = self.train
        bjids = np.ascontiguousarray(tr["bjid"].to_numpy(dtype=np.int32))
        freqs = np.ascontiguousarray(tr["y"].to_numpy(dtype=np.float64))
        f_inter = np.ascontiguousarray(tr[self.partition_cols].to_numpy(dtype=np.float64))
        f_extra = np.ascontiguousarray(tr[self.extra_cols].to_numpy(dtype=np.float64))
        psi = np.array([self.cut, self.cut ** 2 / 2.0], dtype=np.float64)
        return bjids, freqs, f_inter, f_extra, psi


def generate_poisson(cfg: PoissonBenchConfig) -> PoissonBench:
    rng = np.random.default_rng(cfg.seed)
    k = cfg.k_groups
    n_bits = max(1, int(np.ceil(np.log2(k))))
    edges = np.linspace(0.0, 1.0, cfg.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    delta = float(edges[1] - edges[0])

    group = rng.integers(0, k, size=cfg.n_objects)
    bits = ((group[:, None] >> np.arange(n_bits)) & 1).astype(np.float64)
    m = rng.random(cfg.n_objects)

    beta0 = rng.uniform(1.0, 5.0, size=k)
    beta1 = cfg.drift * rng.uniform(-0.8, 0.8, size=k) * beta0

    lam_group = beta0[group][:, None] + beta1[group][:, None] * centers[None, :]
    lam_m = ((3.0 + 2.0 * np.sin(2.0 * np.pi * m))[:, None]
             + cfg.drift * (2.0 * np.cos(2.0 * np.pi * m))[:, None] * centers[None, :])
    lam_unit = np.maximum(cfg.rho * lam_group + (1.0 - cfg.rho) * lam_m, 0.05)
    lam = lam_unit * cfg.exposure   # cohort intensity: events per unit time per object

    counts = np.maximum(rng.poisson(lam * delta), 1)

    n_rows = cfg.n_objects * cfg.n_bins
    obj_idx = np.repeat(np.arange(cfg.n_objects), cfg.n_bins)
    data = {f"f_{i}": bits[obj_idx, i] for i in range(n_bits)}
    data["f_m"] = m[obj_idx]
    data["t"] = np.tile(centers, cfg.n_objects)
    data["e_0"] = np.ones(n_rows)
    data["e_1"] = data["t"]
    data["y"] = counts.reshape(-1).astype(np.float64)
    data["bjid"] = obj_idx.astype(np.int64)
    data["group"] = group[obj_idx].astype(np.int64)
    data["lam_true"] = lam.reshape(-1)

    return PoissonBench(
        df=pd.DataFrame(data),
        partition_cols=[f"f_{i}" for i in range(n_bits)] + ["f_m"],
        extra_cols=["e_0", "e_1"],
        cut=cfg.cut,
        delta=delta,
        config=cfg,
    )


POISSON_PRESETS = {
    "poisson_k8": PoissonBenchConfig(k_groups=8, n_objects=500),
    "poisson_k64": PoissonBenchConfig(k_groups=64, n_objects=2000),
}

__all__ = ["PoissonBenchConfig", "PoissonBench", "generate_poisson", "POISSON_PRESETS"]

"""E1: parametric benchmark family generalizing the paper's MSE/LogLoss generators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchConfig:
    task: str = "mse"            # "mse" | "logloss"
    k_groups: int = 8
    basis: str = "fourier"       # "linear" | "fourier"
    omega: float = 50.0
    rho: float = 1.0             # role purity in [0, 1]
    drift: float = 1.0           # scale of time-dependent coefficient part
    noise: float = 0.01
    cut: float = 0.5
    n_rows: int = 4000
    seed: int = 0


@dataclass
class Bench:
    df: pd.DataFrame
    partition_cols: List[str]
    extra_cols: List[str]
    task: str
    cut: float
    config: Optional[BenchConfig] = None

    @property
    def train(self) -> pd.DataFrame:
        return self.df[self.df["t"] < self.cut]

    @property
    def test(self) -> pd.DataFrame:
        return self.df[self.df["t"] >= self.cut]


def _phi(t: np.ndarray, basis: str, omega: float) -> np.ndarray:
    if basis == "linear":
        return np.column_stack([np.ones_like(t), t])
    if basis == "fourier":
        return np.column_stack([np.ones_like(t), t, np.sin(omega * t), np.cos(omega * t)])
    raise ValueError(f"unknown basis {basis!r}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(cfg: BenchConfig) -> Bench:
    rng = np.random.default_rng(cfg.seed)
    n, k = cfg.n_rows, cfg.k_groups
    n_bits = max(1, int(np.ceil(np.log2(k))))

    t = rng.random(n)
    group = rng.integers(0, k, size=n)
    bits = ((group[:, None] >> np.arange(n_bits)) & 1).astype(np.float64)
    m = rng.random(n)  # mixed nuisance feature: violates the role split for rho < 1

    E = _phi(t, cfg.basis, cfg.omega)
    d = E.shape[1]

    beta = rng.standard_normal((k, d))
    beta[:, 1:] *= cfg.drift                      # non-constant part scaled by drift
    score_pure = np.einsum("ij,ij->i", E, beta[group])

    # impure component: coefficients vary smoothly with m instead of with the group
    beta_m = np.sin(2.0 * np.pi * (np.arange(d)[None, :] + 1.0) * m[:, None])
    beta_m[:, 1:] *= cfg.drift
    score_mixed = np.einsum("ij,ij->i", E, beta_m)

    score = cfg.rho * score_pure + (1.0 - cfg.rho) * score_mixed

    if cfg.task == "mse":
        y = score + cfg.noise * rng.standard_normal(n)
        oracle = score
    elif cfg.task == "logloss":
        p = _sigmoid(score)
        y = (rng.random(n) < p).astype(np.float64)
        oracle = score
    else:
        raise ValueError(f"unknown task {cfg.task!r}")

    data = {f"f_{i}": bits[:, i] for i in range(n_bits)}
    data["f_m"] = m
    data["t"] = t
    for j in range(d):
        data[f"e_{j}"] = E[:, j]
    data["y"] = y
    data["group"] = group.astype(np.int64)
    data["oracle"] = oracle

    df = pd.DataFrame(data)
    return Bench(
        df=df,
        partition_cols=[f"f_{i}" for i in range(n_bits)] + ["f_m"],
        extra_cols=[f"e_{j}" for j in range(d)],
        task=cfg.task,
        cut=cfg.cut,
        config=cfg,
    )


PRESETS = {
    "mse_k8": BenchConfig(task="mse", k_groups=8, basis="fourier", omega=50.0,
                          rho=1.0, drift=1.0, noise=0.01, n_rows=4000),
    "mse_k128": BenchConfig(task="mse", k_groups=128, basis="fourier", omega=50.0,
                            rho=1.0, drift=1.0, noise=0.01, n_rows=10000),
    "regime_base": BenchConfig(task="mse", k_groups=32, basis="fourier", omega=8.0,
                               rho=1.0, drift=1.0, noise=0.05, n_rows=4000),
}

__all__ = ["BenchConfig", "Bench", "generate", "PRESETS"]

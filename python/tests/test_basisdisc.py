import numpy as np
import pandas as pd

from extra_boost_py.experiments.basisdisc import (
    DiscoveredBasis, apply_basis, discover_basis)
from extra_boost_py.experiments.benchgen import Bench


def _bench_from_signal(fn, n=8000, cut=0.75, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    y = fn(t) + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({"f_0": rng.integers(0, 2, n).astype(float),
                       "t": t, "e_0": np.ones(n), "e_1": t, "y": y})
    return Bench(df=df, partition_cols=["f_0"], extra_cols=["e_0", "e_1"],
                 task="mse", cut=cut)


def test_recovers_known_frequency():
    b = _bench_from_signal(lambda t: 0.5 * t + np.sin(2 * np.pi * 8.0 * t))
    basis = discover_basis(b)
    assert len(basis.freqs) >= 1
    assert min(abs(f - 8.0) for f in basis.freqs) < 0.5
    assert basis.r2_train > 0.8


def test_noise_yields_no_frequencies():
    b = _bench_from_signal(lambda t: np.zeros_like(t))
    basis = discover_basis(b)
    assert basis.freqs == ()


def test_leakage_train_window_only():
    # frequency 8 before the cut, 20 after: only 8 may be discovered
    def fn(t):
        return np.where(t < 0.75, np.sin(2 * np.pi * 8.0 * t),
                        np.sin(2 * np.pi * 20.0 * t))
    basis = discover_basis(_bench_from_signal(fn))
    assert len(basis.freqs) >= 1
    assert min(abs(f - 8.0) for f in basis.freqs) < 0.5
    assert all(abs(f - 20.0) > 2.0 for f in basis.freqs)


def test_apply_basis_bookkeeping():
    b = _bench_from_signal(lambda t: np.sin(2 * np.pi * 8.0 * t))
    n_part, n_extra = len(b.partition_cols), len(b.extra_cols)
    basis = DiscoveredBasis(freqs=(8.0,), r2_train=0.9)
    b2 = apply_basis(b, basis)
    assert len(b2.extra_cols) == n_extra + 2
    assert len(b2.partition_cols) == n_part + 2
    assert np.isfinite(b2.df[b2.extra_cols].to_numpy()).all()
    assert np.allclose(b2.df["e_2"], np.sin(2 * np.pi * 8.0 * b2.df["t"]))
    # original untouched
    assert len(b.extra_cols) == n_extra and "e_2" not in b.df.columns

import numpy as np

from extra_boost_py.experiments.poisson_bench import (
    POISSON_PRESETS, PoissonBenchConfig, generate_poisson)


def test_shapes_and_columns():
    b = generate_poisson(PoissonBenchConfig(k_groups=8, n_objects=100, n_bins=10, seed=1))
    assert len(b.df) == 100 * 10
    assert b.partition_cols == ["f_0", "f_1", "f_2", "f_m"]
    assert b.extra_cols == ["e_0", "e_1"]
    assert {"t", "y", "bjid", "group", "lam_true"} <= set(b.df.columns)
    assert (b.df["lam_true"] > 0).all()
    assert (b.df["y"] >= 1).all()  # engine cannot take zero counts
    assert b.task == "poisson"


def test_determinism():
    a = generate_poisson(PoissonBenchConfig(seed=5, n_objects=50))
    b = generate_poisson(PoissonBenchConfig(seed=5, n_objects=50))
    assert a.df.equals(b.df)


def test_counts_match_intensity():
    b = generate_poisson(PoissonBenchConfig(n_objects=2000, n_bins=10, seed=2))
    df = b.df
    # mean count per bin should approximate mean cohort intensity * delta
    ratio = df["y"].mean() / (df["lam_true"].mean() * b.delta)
    assert abs(ratio - 1.0) < 0.05


def test_event_train_consistency():
    b = generate_poisson(PoissonBenchConfig(n_objects=100, seed=3))
    bjids, freqs, f_inter, f_extra, psi = b.event_train()
    assert freqs.sum() == b.train["y"].sum()
    assert f_inter.shape[0] == len(b.train)
    assert np.allclose(psi, [b.cut, b.cut ** 2 / 2.0])


def test_presets():
    assert {"poisson_k8", "poisson_k64"} <= set(POISSON_PRESETS)

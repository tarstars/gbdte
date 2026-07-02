import numpy as np

from extra_boost_py.experiments.benchgen import BenchConfig, generate, PRESETS


def test_shapes_and_columns():
    b = generate(BenchConfig(k_groups=8, n_rows=1000, seed=1))
    assert len(b.df) == 1000
    assert b.partition_cols == ["f_0", "f_1", "f_2", "f_m"]
    assert b.extra_cols == ["e_0", "e_1", "e_2", "e_3"]  # fourier basis
    assert {"t", "y", "group", "oracle"} <= set(b.df.columns)
    assert set(np.unique(b.df["group"])) <= set(range(8))


def test_determinism():
    a = generate(BenchConfig(seed=7, n_rows=500))
    b = generate(BenchConfig(seed=7, n_rows=500))
    assert a.df.equals(b.df)


def test_temporal_split():
    b = generate(BenchConfig(n_rows=1000, cut=0.5, seed=0))
    assert (b.train["t"] < 0.5).all() and (b.test["t"] >= 0.5).all()
    assert len(b.train) + len(b.test) == 1000


def test_rho1_pure_roles():
    # at rho=1 target is exactly phi(t)^T beta_g + noise: per-group regression on the
    # e-columns must fit near-perfectly
    b = generate(BenchConfig(rho=1.0, noise=0.0, n_rows=2000, seed=3))
    df = b.df
    for g in range(4):
        part = df[df["group"] == g]
        E = part[b.extra_cols].to_numpy()
        coef, *_ = np.linalg.lstsq(E, part["y"].to_numpy(), rcond=None)
        resid = part["y"].to_numpy() - E @ coef
        assert np.abs(resid).max() < 1e-8


def test_logloss_has_bernoulli_target_and_oracle():
    b = generate(BenchConfig(task="logloss", noise=0.0, n_rows=1000, seed=2))
    assert set(np.unique(b.df["y"])) <= {0.0, 1.0}
    assert np.isfinite(b.df["oracle"]).all()


def test_presets_exist():
    assert {"mse_k8", "mse_k128", "regime_base"} <= set(PRESETS)

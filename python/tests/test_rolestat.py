import numpy as np

from extra_boost_py.experiments.benchgen import BenchConfig, generate
from extra_boost_py.experiments.rolestat import (
    auto_roles, drift_score, separability_index)


def test_drift_score_detects_time_feature():
    rng = np.random.default_rng(0)
    t = rng.random(4000)
    assert drift_score(t + 0.1 * rng.standard_normal(4000), t) > 0.9
    assert drift_score(rng.random(4000), t) < 0.55


def test_separability_monotone_in_rho():
    vals = []
    for rho in (0.0, 0.5, 1.0):
        idx = np.mean([
            separability_index(generate(BenchConfig(
                k_groups=8, rho=rho, drift=1.0, omega=8.0,
                noise=0.05, n_rows=4000, seed=s)))
            for s in range(3)
        ])
        vals.append(idx)
    assert vals[0] < vals[1] < vals[2]


def test_auto_roles_recovers_design():
    b = generate(BenchConfig(k_groups=8, rho=1.0, omega=8.0, n_rows=4000, seed=0))
    part, leaf = auto_roles(b)
    assert "t" in leaf
    assert {"f_0", "f_1", "f_2"} <= set(part)


def test_separability_sees_categorical_group():
    import pandas as pd
    from extra_boost_py.experiments.benchgen import Bench
    rng = np.random.default_rng(0)
    n = 6000
    g = rng.integers(0, 10, n)                 # 10-way categorical (not binary)
    t = rng.random(n)
    y = g.astype(float) + 0.3 * rng.standard_normal(n)   # stable group->y map over time
    df = pd.DataFrame({"g": g.astype(float), "t": t, "e_0": np.ones(n), "e_1": t, "y": y})
    b = Bench(df=df, partition_cols=["g"], extra_cols=["e_0", "e_1"], task="mse", cut=0.5)
    assert separability_index(b) > 0.5          # old binary-only code returned 0.0


def test_separability_still_monotone_in_rho():
    from dataclasses import replace
    from extra_boost_py.experiments.benchgen import PRESETS, generate
    base = PRESETS["regime_base"]
    vals = [float(np.mean([separability_index(generate(replace(base, rho=r, seed=s)))
                           for s in range(3)])) for r in (0.0, 0.5, 1.0)]
    assert vals[0] < vals[1] < vals[2]


def _panel_bench(with_slope, seed=0, periodic=False):
    import pandas as pd
    from extra_boost_py.experiments.benchgen import Bench
    rng = np.random.default_rng(seed)
    n = 6000
    g = rng.integers(0, 6, n)
    t = rng.random(n)
    slope = (g - 2.5) if with_slope else 0.0          # per-group DIFFERENT slopes
    periodic_term = np.sin(2 * np.pi * 6 * t) if periodic else 0.0
    y = g.astype(float) + slope * t + periodic_term + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({"g": g.astype(float), "t": t, "e_0": np.ones(n), "e_1": t, "y": y})
    return Bench(df=df, partition_cols=["g"], extra_cols=["e_0", "e_1"], task="mse", cut=0.5)


def test_extrapolation_gain_high_with_per_group_slopes():
    from extra_boost_py.experiments.rolestat import extrapolation_gain
    assert extrapolation_gain(_panel_bench(with_slope=True)) > 0.2


def test_extrapolation_gain_zero_without_slopes():
    from extra_boost_py.experiments.rolestat import extrapolation_gain
    assert extrapolation_gain(_panel_bench(with_slope=False)) < 0.05


def test_extrapolation_gain_periodic_less_than_trend():
    from extra_boost_py.experiments.rolestat import extrapolation_gain
    g_trend = extrapolation_gain(_panel_bench(with_slope=True))
    g_periodic = extrapolation_gain(_panel_bench(with_slope=False, periodic=True))
    assert g_periodic < g_trend

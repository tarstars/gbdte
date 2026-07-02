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

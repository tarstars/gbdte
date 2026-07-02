import numpy as np
from sklearn.metrics import log_loss

from extra_boost_py.experiments.classical_bench import classical_bench


def test_structure():
    b = classical_bench(n_rows=2000, seed=0)
    assert b.task == "logloss"
    assert len(b.partition_cols) == 17 and b.extra_cols == ["e_0", "e_1"]
    assert (b.train["t"] < 0.4).all() and (b.test["t"] >= 0.4).all()


def test_oracle_beats_marginal():
    b = classical_bench(n_rows=20000, seed=1)
    y = b.df["y"].to_numpy()
    p_oracle = 1.0 / (1.0 + np.exp(-b.df["oracle"].to_numpy()))
    ll_oracle = log_loss(y, p_oracle)
    ll_marginal = log_loss(y, np.full_like(y, y.mean()))
    assert ll_oracle < ll_marginal - 0.02

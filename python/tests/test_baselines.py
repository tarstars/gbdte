import numpy as np
import pytest

from extra_boost_py.experiments.baselines import make_models
from extra_boost_py.experiments.benchgen import BenchConfig, generate


@pytest.mark.parametrize("task", ["mse", "logloss"])
def test_all_models_fit_predict(task):
    b = generate(BenchConfig(task=task, k_groups=4, omega=8.0, n_rows=600, seed=0))
    models = make_models(task)
    assert "gbdte" in models and "lgbm_linear" in models
    for name, model in models.items():
        if not model.available():
            continue
        model.fit(b)
        pred = model.predict(b.test)
        assert pred.shape == (len(b.test),)
        assert np.isfinite(pred).all(), name

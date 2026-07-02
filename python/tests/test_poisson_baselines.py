import numpy as np

from extra_boost_py.experiments.poisson_baselines import make_poisson_models
from extra_boost_py.experiments.poisson_bench import PoissonBenchConfig, generate_poisson
from extra_boost_py.experiments.stats import evaluate


def test_all_poisson_models_fit_predict():
    b = generate_poisson(PoissonBenchConfig(k_groups=4, n_objects=150, n_bins=8, seed=0))
    models = make_poisson_models()
    assert {"gbdte_pois", "lgbm_pois", "oracle_pois"} <= set(models)
    for name, model in models.items():
        if not model.available():
            continue
        model.fit(b)
        pred = model.predict(b.test)
        assert pred.shape == (len(b.test),)
        assert np.isfinite(pred).all() and (pred >= 0).all(), name


def test_gbdte_pois_recovers_intensity():
    b = generate_poisson(PoissonBenchConfig(k_groups=4, n_objects=400, n_bins=10,
                                            drift=1.0, seed=1))
    models = make_poisson_models()
    m = models["gbdte_pois"]
    m.fit(b, {"max_depth": 3})
    res = evaluate(m, b)
    oracle = models["oracle_pois"]
    oracle.fit(b)
    res_oracle = evaluate(oracle, b)
    # designed-roles GBDTE should land close to the oracle's deviance
    assert res["poisson_dev"] < 1.5 * res_oracle["poisson_dev"]


def test_evaluate_poisson_oracle_beats_mean():
    b = generate_poisson(PoissonBenchConfig(k_groups=8, n_objects=800, n_bins=10, seed=1))
    models = make_poisson_models()
    oracle = models["oracle_pois"]
    oracle.fit(b)
    res_oracle = evaluate(oracle, b)
    assert set(res_oracle) == {"poisson_dev", "rate_rmse"}

    class MeanModel:
        name = "mean"
        def available(self):
            return True
        def fit(self, bench, params=None):
            self._mu = bench.train["y"].mean()
        def predict(self, df):
            return np.full(len(df), self._mu)

    mm = MeanModel()
    mm.fit(b)
    res_mean = evaluate(mm, b)
    assert res_oracle["poisson_dev"] < res_mean["poisson_dev"]
    assert res_oracle["rate_rmse"] < 1e-9

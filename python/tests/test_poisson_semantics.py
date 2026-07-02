"""Empirical validation of the legacy Poisson engine's semantics (gate for the
Poisson experiment layer).

Validated semantics (2026-07-02):
- rows are (object/cohort, time-bin) observations; `freqs` are per-bin event COUNTS and
  must be strictly positive (zero counts crash the CGO layer);
- `features_extra` = phi(t) at bin centers, `psi` = integral of phi over the full
  observation window;
- `predict` returns the expected per-bin count, i.e. lambda(t) in freq units;
- ONLY n_stages=1 is usable in extra mode: the first tree fits leaf intensities exactly
  (learning_rate is ignored for it), while every additional stage adds a systematic
  offset (bias mis-propagation bug in wholeLossNextTreeExtra) — see xfail test below.
"""
import numpy as np
import pytest

from extra_boost_py.poisson_booster import PoissonLegacyBooster, PoissonLegacyParams


def test_basic_mode_recovers_group_rates():
    # two groups, constant rates 10 and 30, one row per object
    rng = np.random.default_rng(0)
    n = 400
    grp = rng.integers(0, 2, n)
    freqs = rng.poisson(np.where(grp == 0, 10.0, 30.0)).astype(np.float64)
    bjids = np.arange(n, dtype=np.int32)
    f_inter = grp.reshape(-1, 1).astype(np.float64)
    booster = PoissonLegacyBooster.train(
        bjids=bjids, freqs=freqs, features_inter=f_inter,
        params=PoissonLegacyParams(n_stages=1, max_depth=1, learning_rate=1.0))
    preds = booster.predict(np.array([[0.0], [1.0]]))
    assert preds[0] == pytest.approx(freqs[grp == 0].mean(), rel=0.05)
    assert preds[1] == pytest.approx(freqs[grp == 1].mean(), rel=0.05)


def _binned_two_group_data(rng, n_obj=200, n_bins=10, exposure=20):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    delta = edges[1] - edges[0]
    rows = []
    for j in range(n_obj):
        g = j % 2
        lam = 2.0 + 2.0 * centers if g == 0 else 4.0 - 2.0 * centers
        counts = np.maximum(rng.poisson(lam * delta * exposure), 1)
        for b in range(n_bins):
            rows.append((j, counts[b], float(g), centers[b]))
    arr = np.array(rows)
    return arr, delta, exposure


def _train(arr, n_stages):
    return PoissonLegacyBooster.train(
        bjids=arr[:, 0].astype(np.int32), freqs=arr[:, 1],
        features_inter=arr[:, 2].reshape(-1, 1),
        features_extra=np.column_stack([np.ones(len(arr)), arr[:, 3]]),
        psi=np.array([1.0, 0.5]),
        params=PoissonLegacyParams(n_stages=n_stages, max_depth=3,
                                   learning_rate=0.2, reg_lambda=1e-8))


def test_extra_mode_single_stage_recovers_linear_intensity():
    rng = np.random.default_rng(0)
    arr, delta, exposure = _binned_two_group_data(rng)
    booster = _train(arr, n_stages=1)
    probe_t = np.array([0.1, 0.5, 0.9])
    fe = np.column_stack([np.ones(3), probe_t])
    for g, lam_fn in [(0.0, lambda tt: 2.0 + 2.0 * tt), (1.0, lambda tt: 4.0 - 2.0 * tt)]:
        preds = booster.predict(np.full((3, 1), g), fe)
        rates = preds / (exposure * delta)  # predictions are per-bin counts
        assert np.allclose(rates, lam_fn(probe_t), rtol=0.06), (rates, lam_fn(probe_t))


@pytest.mark.xfail(reason="engine bug: each extra-mode stage beyond the first adds a "
                          "systematic offset (bias mis-propagation); fix in Go before "
                          "using n_stages > 1", strict=True)
def test_extra_mode_multi_stage_is_consistent():
    rng = np.random.default_rng(0)
    arr, delta, exposure = _binned_two_group_data(rng)
    booster = _train(arr, n_stages=2)
    probe_t = np.array([0.1, 0.9])
    fe = np.column_stack([np.ones(2), probe_t])
    preds = booster.predict(np.zeros((2, 1)), fe) / (exposure * delta)
    assert np.allclose(preds, 2.0 + 2.0 * probe_t, rtol=0.06)

import numpy as np

from extra_boost_py import PoissonLegacyBooster, PoissonLegacyParams


def run_basic():
    bjids = np.array([101, 102, 103, 104], dtype=np.int32)
    freqs = np.array([10, 30, 10, 30], dtype=np.float64)
    features_inter = np.array([[1, 1], [2, 3], [3, 2], [4, 4]], dtype=np.float64)

    params = PoissonLegacyParams(n_stages=1, max_depth=1, learning_rate=1.0, unbalanced_penalty=0.1)
    booster = PoissonLegacyBooster.train(
        bjids=bjids,
        freqs=freqs,
        features_inter=features_inter,
        params=params,
    )

    features_test = np.array([[1, -1], [2, 10], [3, 0], [4, 5], [5, 2]], dtype=np.float64)
    preds = booster.predict(features_test)
    print("basic preds:", preds)


def run_extra():
    bjids = np.array([101, 101, 102, 102, 103, 103, 104, 104], dtype=np.int32)
    freqs = np.array([10, 30, 50, 70, 30, 50, 70, 90], dtype=np.float64)
    features_inter = np.array(
        [
            [1, 1, 4],
            [1, 1, 4],
            [1, 4, 1],
            [1, 4, 1],
            [3, 2, 3],
            [3, 2, 3],
            [3, 3, 2],
            [3, 3, 2],
        ],
        dtype=np.float64,
    )
    time = np.array([0.0, 0.2, 0.4, 0.6, 0.0, 0.2, 0.4, 0.6], dtype=np.float64)
    features_extra = np.stack([np.ones_like(time), time], axis=1)
    t_range = np.linspace(0.0, 1.0, 100)
    dt = t_range[1] - t_range[0]
    psi = np.array([np.sum(np.ones_like(t_range)) * dt, np.sum(t_range) * dt], dtype=np.float64)

    params = PoissonLegacyParams(n_stages=1, max_depth=1, learning_rate=0.2, reg_lambda=1e-8)
    booster = PoissonLegacyBooster.train(
        bjids=bjids,
        freqs=freqs,
        features_inter=features_inter,
        features_extra=features_extra,
        psi=psi,
        params=params,
    )
    preds = booster.predict(features_inter, features_extra)
    print("extra preds:", preds)


if __name__ == "__main__":
    run_basic()
    run_extra()

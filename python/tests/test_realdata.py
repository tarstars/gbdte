import numpy as np
import pandas as pd

from extra_boost_py.experiments.realdata import (
    REAL_DATASETS, RealDatasetSpec, frame_to_bench)


def _toy_spec():
    return RealDatasetSpec(name="toy", kaggle_ref="x/y", kaggle_kind="dataset",
                           files=("f.parquet",), time_col="ts", target_col="tgt",
                           task="mse")


def _toy_frame(n=1000):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=n, freq="h")[rng.permutation(n)],
        "tgt": rng.standard_normal(n),
        "num": rng.standard_normal(n),
        "num_nan": np.where(rng.random(n) < 0.3, np.nan, rng.standard_normal(n)),
        "cat": rng.choice(["a", "b", None], n),
    })


def test_transform_shapes_and_columns():
    b = frame_to_bench(_toy_frame(), _toy_spec(), seed=0, n_max=500)
    assert len(b.df) == 500
    assert b.extra_cols == ["e_0", "e_1"]
    assert set(b.partition_cols) == {"num", "num_nan", "cat"}
    assert b.task == "mse"
    assert (b.df["e_0"] == 1.0).all()
    assert np.allclose(b.df["e_1"], b.df["t"])
    assert b.df["t"].between(0, 1).all()
    assert not b.df[b.partition_cols].isna().any().any()   # NaNs imputed/encoded
    assert b.df[b.partition_cols].dtypes.apply(
        lambda d: np.issubdtype(d, np.number)).all()        # all numeric


def test_temporal_split_at_quantile():
    b = frame_to_bench(_toy_frame(), _toy_spec(), seed=1, n_max=1000)
    assert abs(len(b.train) / len(b.df) - 0.75) < 0.02
    assert b.train["t"].max() <= b.test["t"].min() + 1e-9


def test_seeded_subsample_differs():
    a = frame_to_bench(_toy_frame(), _toy_spec(), seed=0, n_max=300)
    c = frame_to_bench(_toy_frame(), _toy_spec(), seed=1, n_max=300)
    assert not a.df.reset_index(drop=True).equals(c.df.reset_index(drop=True))


def test_registry():
    assert {"weather", "sberbank", "homecredit"} <= set(REAL_DATASETS)
    assert REAL_DATASETS["homecredit"].task == "logloss"

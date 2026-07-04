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


def test_latam_aggregation():
    import pandas as pd
    from extra_boost_py.experiments.realdata import _latam_frame

    card = pd.DataFrame({
        "customer_id": [1, 2],
        "age": [30, 40], "gender": ["M", "F"], "location": ["a", "b"],
        "income_bracket": ["High", "Low"], "occupation": ["x", "y"],
        "education_level": ["e1", "e2"], "marital_status": ["m", "s"],
        "household_size": [2, 3], "acquisition_channel": ["c1", "c2"],
        "customer_segment": ["s1", "s2"], "savings_account": [True, False],
        "credit_card": [True, True], "personal_loan": [False, False],
        "investment_account": [True, False], "insurance_product": [False, True],
        "active_products": [2, 1],
        "tx_count": [99, 99], "churn_probability": [0.5, 0.5],  # leaky: must be dropped
    })
    tx = pd.DataFrame({
        "customer_id": [1, 1, 1, 2],
        "date": ["2023-01-05", "2023-01-20", "2023-03-02", "2023-06-10"],
        "amount": [10.0, 20.0, 30.0, 40.0],
        "type": ["Transfer", "Payment", "Transfer", "Withdrawal"],
    })
    frame = _latam_frame(card, tx)
    assert len(frame) == 2 * 12                       # full client x month grid
    assert "tx_count" not in frame.columns            # leaky aggregate excluded
    assert "churn_probability" not in frame.columns
    jan1 = frame[(frame["customer_id"] == 1) & (frame["month"] == "2023-01-01")]
    feb1 = frame[(frame["customer_id"] == 1) & (frame["month"] == "2023-02-01")]
    assert float(jan1["y_log_count"].iloc[0]) == pytest_approx_log(2)
    assert float(feb1["y_log_count"].iloc[0]) == 0.0  # zero month materialized
    assert "age" in frame.columns and "customer_id" in frame.columns


def pytest_approx_log(n):
    import numpy as np
    return float(np.log1p(n))


def test_frame_to_bench_numeric_time():
    import pandas as pd
    from extra_boost_py.experiments.realdata import RealDatasetSpec, frame_to_bench
    df = pd.DataFrame({"batch": np.repeat(np.arange(1, 11), 20).astype(float),
                       "y": np.random.default_rng(0).standard_normal(200),
                       "s1": np.random.default_rng(1).standard_normal(200)})
    spec = RealDatasetSpec(name="x", kaggle_ref="", kaggle_kind="uci", files=(),
                           time_col="batch", target_col="y", task="logloss")
    b = frame_to_bench(df, spec, seed=0, n_max=200)
    assert b.df["t"].between(0, 1).all()
    assert "s1" in b.partition_cols and "batch" not in b.partition_cols


def test_parse_gas_batch():
    import tempfile, os
    from extra_boost_py.experiments.realdata import _parse_gas_batch
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as f:
        f.write("1;10.0 1:0.5 2:-0.25 3:1.0\n6;5.0 1:0.1 2:0.2 3:0.3\n")
        path = f.name
    rows = _parse_gas_batch(path, batch_idx=3)
    os.unlink(path)
    assert len(rows) == 2
    assert rows[0]["gas"] == 1 and rows[0]["batch"] == 3
    assert rows[0]["s1"] == 0.5 and rows[0]["s2"] == -0.25
    assert "s0" not in rows[1]  # concentration token dropped

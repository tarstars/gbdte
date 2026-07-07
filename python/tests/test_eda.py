import pathlib

import pandas as pd
import pytest

from extra_boost_py.experiments.eda import RAW_INFO, REAL_ORDER, load_raw


def test_registry_covers_all_real():
    assert set(REAL_ORDER) <= set(RAW_INFO)
    assert "owid_childmort" in RAW_INFO and RAW_INFO["owid_childmort"].source.startswith("http")


def test_load_raw_gassensor_has_real_columns():
    if not pathlib.Path("datasets/realdata/gassensor/extract.parquet").exists():
        pytest.skip("cache absent")
    df = load_raw("gassensor")
    assert "gas" in df.columns and "s1" in df.columns          # gas class + sensor features
    assert df["gas"].nunique() >= 2


def test_load_raw_owid_has_country_names():
    df = load_raw("owid_childmort")                            # network re-fetch (cached)
    assert {"entity", "year"} <= set(df.columns)
    assert pd.api.types.is_string_dtype(df["entity"]) and (df["entity"] == "India").any()


import plotly.graph_objects as go
from extra_boost_py.experiments.eda import (
    apply_style, column_table, distribution_fig, target_over_time_fig, overview_metrics)


def test_column_table_shape():
    df = load_raw("gassensor")
    ct = column_table(df, "gassensor")
    assert list(ct.columns) == ["column", "dtype", "missing_%", "n_unique", "example", "meaning"]
    assert len(ct) == df.shape[1]


def test_figs_are_plotly():
    df = load_raw("gassensor")
    assert isinstance(distribution_fig(df, "s1"), go.Figure)
    assert isinstance(target_over_time_fig("owid_childmort"), go.Figure)
    assert isinstance(apply_style(go.Figure()), go.Figure)


def test_overview_metrics_keys():
    m = overview_metrics("owid_childmort")
    assert {"records", "time_span", "task", "train", "test"} <= set(m)

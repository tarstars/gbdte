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

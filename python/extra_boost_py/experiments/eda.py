"""EDA layer for the experiments notebook: raw loaders + per-dataset metadata.
Kept pure and importable so the notebook stays a thin narrative."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import realdata as R

REAL_ORDER = ["owid_childmort", "owid_maternalmort", "owid_lifeexp", "owid_gdppcap",
              "weather", "sberbank", "homecredit", "latam", "gassensor"]
SYNTH_ORDER = ["mse", "logloss", "poisson"]


@dataclass
class DatasetInfo:
    description: str
    source: str
    task: str
    notes: str = ""
    column_meanings: Dict[str, str] = field(default_factory=dict)


RAW_INFO: Dict[str, DatasetInfo] = {
    "owid_childmort": DatasetInfo(
        "Under-five deaths per 100 live births, by country and year.",
        "https://ourworldindata.org/grapher/child-mortality", "regression (log rate)",
        "Declines ~exponentially; the Lee-Carter per-group linear-in-year extrapolation problem.",
        {"entity": "country name", "code": "ISO-3 code", "year": "calendar year",
         "value": "child mortality rate (deaths per 100 live births)"}),
    "owid_maternalmort": DatasetInfo(
        "Maternal deaths per 100,000 live births, by country and year.",
        "https://ourworldindata.org/grapher/maternal-mortality", "regression (log rate)",
        "Strong secular decline; per-country slopes differ.",
        {"entity": "country", "year": "year", "value": "maternal mortality ratio"}),
    "owid_lifeexp": DatasetInfo(
        "Period life expectancy at birth, by country and year.",
        "https://ourworldindata.org/grapher/life-expectancy", "regression",
        "Bounded, ~linear rise; no log transform.",
        {"entity": "country", "year": "year", "value": "life expectancy (years)"}),
    "owid_gdppcap": DatasetInfo(
        "GDP per capita (PPP, constant intl-$), by country and year.",
        "https://ourworldindata.org/grapher/gdp-per-capita-worldbank", "regression (log)",
        "Grows ~exponentially.",
        {"entity": "country", "year": "year", "value": "GDP per capita, PPP"}),
    "weather": DatasetInfo(
        "TabReD Weather: numerical weather-station readings; predict temperature.",
        "https://github.com/yandex-research/tabred", "regression",
        "Strong diurnal/seasonal PERIODICITY -> a negative for GBDTE (a periodic basis "
        "converts extrapolation into interpolation).",
        {"fact_temperature": "observed temperature (target)", "fact_time": "timestamp",
         "gfs_*": "GFS model forecast features", "cmc_*": "CMC model forecast features"}),
    "sberbank": DatasetInfo(
        "Sberbank Russian housing: property transactions 2011-2015; predict sale price.",
        "https://www.kaggle.com/competitions/sberbank-russian-housing-market", "regression",
        "Weak stable group structure; modest trend.",
        {"timestamp": "transaction date", "price_doc": "sale price (target)",
         "full_sq": "total area m^2", "life_sq": "living area m^2"}),
    "homecredit": DatasetInfo(
        "Home Credit default risk: loan applications; predict default.",
        "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability",
        "classification", "The credit-scoring domain the abstract names; no per-group forward trend.",
        {"date_decision": "application date", "target": "default flag (target)"}),
    "latam": DatasetInfo(
        "LatAm fintech: 48k clients, 12 months of transactions; predict monthly activity.",
        "https://data.mendeley.com/datasets/mhb4zn3258/1", "regression (log count)",
        "Static client card does not determine activity level (low separability).",
        {"customer_id": "client id", "month": "calendar month",
         "y_log_count": "log(1+monthly tx count) (target)",
         "age": "client age", "income_bracket": "income band"}),
    "gassensor": DatasetInfo(
        "UCI Gas Sensor Array Drift: 16 chemo-sensors x 8 features over 10 time batches; "
        "one-vs-rest gas classification.",
        "https://archive.ics.uci.edu/dataset/224", "classification",
        "Sensors drift over time, but no per-class forward TREND to extrapolate.",
        {"gas": "gas class 1-6", "batch": "time batch 1-10 (36 months)",
         "s1..s128": "16 sensors x 8 response features"}),
}

_OWID_SLUGS = {"owid_childmort": ("child-mortality", "child_mortality_rate"),
               "owid_maternalmort": ("maternal-mortality", "mmr"),
               "owid_lifeexp": ("life-expectancy", "life_expectancy_0"),
               "owid_gdppcap": ("gdp-per-capita-worldbank", "ny_gdp_pcap_pp_kd")}


def load_raw(name: str, n_max: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Raw frame with ORIGINAL columns and real values (country names etc.)."""
    if name in _OWID_SLUGS:
        cache = Path("datasets/realdata") / name / "raw_eda.parquet"
        if not cache.exists():
            slug, valcol = _OWID_SLUGS[name]
            df = R._owid_csv(f"https://ourworldindata.org/grapher/{slug}.csv"
                             f"?csvType=full&useColumnShortNames=true")
            df = df.rename(columns={valcol: "value"})
            keep = [c for c in ["entity", "code", "year", "value"] if c in df.columns]
            df = df[keep].dropna(subset=["value"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache)
        return pd.read_parquet(cache)
    df = pd.read_parquet(Path("datasets/realdata") / name / "extract.parquet")
    if len(df) > n_max:
        df = df.iloc[np.random.default_rng(seed).choice(len(df), n_max, replace=False)]
    return df.reset_index(drop=True)


__all__: List[str] = ["DatasetInfo", "RAW_INFO", "REAL_ORDER", "SYNTH_ORDER", "load_raw"]

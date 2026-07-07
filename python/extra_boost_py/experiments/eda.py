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


import plotly.graph_objects as go

from .realdata import load_real
from .rolestat import extrapolation_gain, separability_index

# Colours: the validated dataviz reference palette in fixed slot order. GBDTE = blue
# highlight; boosters recede in muted/warm hues.
PALETTE = {"gbdte": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "red": "#e34948",
           "violet": "#4a3aa7", "orange": "#eb6834", "muted": "#8a8a86",
           "ink": "#0b0b0b", "grid": "#e6e6e2", "surface": "#fcfcfb", "mid": "#f0efec"}
MODEL_COLOR = {"gbdte_auto": PALETTE["gbdte"], "gbdte": PALETTE["gbdte"],
               "gbdte_const": PALETTE["violet"], "detrend_lgbm": PALETTE["aqua"],
               "lgbm": PALETTE["muted"], "xgb": PALETTE["yellow"],
               "catboost": PALETTE["orange"], "lgbm_linear": PALETTE["red"]}


def apply_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="plotly_white", font=dict(color=PALETTE["ink"], size=13),
                      paper_bgcolor=PALETTE["surface"], plot_bgcolor=PALETTE["surface"],
                      margin=dict(l=60, r=20, t=50, b=50), hovermode="closest",
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False)
    return fig


def column_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    meanings = RAW_INFO[name].column_meanings if name in RAW_INFO else {}

    def mean_of(c: str) -> str:
        if c in meanings:
            return meanings[c]
        for pat, txt in meanings.items():          # wildcard like "gfs_*" or "s1..s128"
            if pat.endswith("*") and c.startswith(pat[:-1]):
                return txt
        if "s1..s128" in meanings and c.startswith("s") and c[1:].isdigit():
            return meanings["s1..s128"]
        return ""

    rows = []
    for c in df.columns:
        s = df[c]
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        rows.append({"column": c, "dtype": str(s.dtype),
                     "missing_%": round(100 * float(s.isna().mean()), 1),
                     "n_unique": int(s.nunique()), "example": ex, "meaning": mean_of(c)})
    return pd.DataFrame(rows)[["column", "dtype", "missing_%", "n_unique", "example", "meaning"]]


def distribution_fig(df: pd.DataFrame, col: str) -> go.Figure:
    s = df[col].dropna()
    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
        fig.add_histogram(x=np.asarray(s, dtype=float), marker_color=PALETTE["gbdte"], nbinsx=40)
    else:
        vc = s.astype(str).value_counts().head(15)[::-1]
        fig.add_bar(x=vc.values, y=vc.index.tolist(), orientation="h", marker_color=PALETTE["gbdte"])
    fig.update_layout(title=f"distribution of {col}")
    return apply_style(fig)


def _time_col(name: str) -> str:
    return {"weather": "fact_time", "sberbank": "timestamp", "homecredit": "date_decision",
            "latam": "month"}.get(name, "year")


def target_over_time_fig(name: str) -> go.Figure:
    """Target vs time; for OWID a few real countries, else binned mean with split line."""
    b = load_real(name, seed=0, n_max=20000)
    fig = go.Figure()
    if name in _OWID_SLUGS:
        raw = load_raw(name)
        for ent, key in zip(["India", "Brazil", "Nigeria", "Germany", "Japan"],
                            ["gbdte", "aqua", "orange", "violet", "red"]):
            e = raw[raw["entity"] == ent].sort_values("year")
            if len(e):
                fig.add_scatter(x=e["year"], y=e["value"], mode="lines", name=ent,
                                line=dict(color=PALETTE[key], width=2))
        fig.update_layout(title=f"{name}: target trajectories (sample countries)",
                          xaxis_title="year",
                          yaxis_title=RAW_INFO[name].column_meanings.get("value", "value"))
    else:
        df = b.df
        g = (df.groupby(pd.cut(df["t"], 40, labels=False))
               .agg(t=("t", "mean"), y=("y", "mean")).dropna())
        fig.add_scatter(x=g["t"], y=g["y"], mode="lines",
                        line=dict(color=PALETTE["gbdte"], width=2), name="mean target")
        fig.add_vline(x=b.cut, line=dict(color=PALETTE["red"], dash="dash"))
        fig.update_layout(title=f"{name}: mean target over normalised time",
                          xaxis_title="t", yaxis_title="y")
    return apply_style(fig)


def overview_metrics(name: str) -> dict:
    b = load_real(name, seed=0, n_max=20000)
    raw = load_raw(name)
    tcol = _time_col(name)
    span = ""
    if tcol in raw.columns:
        try:
            ts = pd.to_datetime(raw[tcol]) if raw[tcol].dtype == object else raw[tcol]
            span = f"{ts.min()} … {ts.max()}"
        except Exception:
            span = f"{raw[tcol].min()} … {raw[tcol].max()}"
    return {"records": len(raw), "time_span": span, "task": b.task,
            "features": len(b.partition_cols), "train": len(b.train), "test": len(b.test)}


def diagnostic_verdict(name: str) -> dict:
    b = load_real(name, seed=0, n_max=20000)
    eg = extrapolation_gain(b)
    return {"separability": round(separability_index(b), 3), "extrap_gain": round(eg, 3),
            "promoted": bool(eg >= 0.05),
            "why": ("green: per-group forward trend headroom" if eg >= 0.05
                    else "red: no per-group forward trend to extrapolate")}


__all__: List[str] = ["DatasetInfo", "RAW_INFO", "REAL_ORDER", "SYNTH_ORDER", "load_raw",
                      "PALETTE", "MODEL_COLOR", "apply_style", "column_table",
                      "distribution_fig", "target_over_time_fig", "overview_metrics",
                      "diagnostic_verdict"]

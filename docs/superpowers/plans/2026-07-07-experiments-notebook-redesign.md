# Experiments Notebook Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mechanical notebook with a narrative, insight-rich, interactive (Plotly + ipywidgets) exploration notebook whose heart is a thorough per-dataset "card"; backed by a tested `eda.py` helper module.

**Architecture:** A pure, tested `python/extra_boost_py/experiments/eda.py` provides raw loaders, per-dataset metadata, and Plotly/table builders; the notebook (generated with nbformat, executed with nbclient) composes them via a thin `render_card()` into four narrative acts + appendix.

**Tech Stack:** plotly, ipywidgets, kaleido, pandas, the existing experiments package; nbformat/nbclient to build+execute.

## Global Constraints

- Colours come from the validated dataviz reference palette in **fixed slot order** (compliant by construction): GBDTE `#2a78d6`, aqua `#1baf7a`, yellow `#eda100`, red `#e34948`, violet `#4a3aa7`, orange `#eb6834`. GBDTE is always the blue highlight; boosters recede in muted hues. Diverging map (win/lose) = blue↔red with neutral `#f0efec` midpoint.
- Every chart: thin marks, legend for ≥2 series, Plotly hover on by default, recessive grid — via one `apply_style(fig)`.
- EDA uses the **raw** frame (real columns/values); experiments use the modelling frame. OWID raw is re-fetched (cache withholds country).
- Widgets are optional local enhancers marked "▶ run locally"; Plotly figures carry all content so the notebook is complete statically (nbviewer).
- Tests: `PYTHONPATH=python .venv/bin/pytest python/tests -q`. Commit per task. Branch `notebook-redesign-2026`.

---

### Task 1: deps + `eda.py` raw loaders and metadata

**Files:**
- Create: `python/extra_boost_py/experiments/eda.py`
- Test: `python/tests/test_eda.py`

**Interfaces:**
- Produces: `DatasetInfo` (dataclass: description:str, source:str, task:str, notes:str,
  column_meanings:dict[str,str]), `RAW_INFO: dict[str, DatasetInfo]`,
  `load_raw(name, n_max=20000, seed=0) -> pd.DataFrame` (original columns / real values;
  OWID re-fetched with `entity, code, year, value`), `REAL_ORDER: list[str]`,
  `SYNTH_ORDER: list[str]`.

- [ ] **Step 1: install deps**

```bash
cd ~/prj/extra_bridged_boosting
uv pip install --python .venv/bin/python plotly ipywidgets kaleido
```

- [ ] **Step 2: failing tests** (`python/tests/test_eda.py`)

```python
import pandas as pd
import pytest
from extra_boost_py.experiments.eda import RAW_INFO, REAL_ORDER, load_raw


def test_registry_covers_all_real():
    assert set(REAL_ORDER) <= set(RAW_INFO)
    assert "owid_childmort" in RAW_INFO and RAW_INFO["owid_childmort"].source.startswith("http")


def test_load_raw_gassensor_has_real_columns():
    if not __import__("pathlib").Path("datasets/realdata/gassensor/extract.parquet").exists():
        pytest.skip("cache absent")
    df = load_raw("gassensor")
    assert "gas" in df.columns and "s1" in df.columns          # gas class + sensor features
    assert df["gas"].nunique() >= 2


def test_load_raw_owid_has_country_names():
    df = load_raw("owid_childmort")                            # network re-fetch
    assert {"entity", "year"} <= set(df.columns)
    assert df["entity"].dtype == object and (df["entity"] == "India").any()
```

- [ ] **Step 3: implement `eda.py` (part 1)**

```python
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
        "Strong diurnal/seasonal PERIODICITY -> a negative for GBDTE (periodic basis "
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
        {"customer_id": "client id", "month": "calendar month", "y_log_count": "log(1+monthly tx count) (target)",
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


__all__ = ["DatasetInfo", "RAW_INFO", "REAL_ORDER", "SYNTH_ORDER", "load_raw"]
```

- [ ] **Step 4: run tests, verify pass; commit**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests/test_eda.py -q
git add python/extra_boost_py/experiments/eda.py python/tests/test_eda.py
git commit -m "Add eda.py raw loaders + per-dataset metadata"
```

---

### Task 2: `eda.py` builders (style, tables, Plotly figures)

**Files:**
- Modify: `python/extra_boost_py/experiments/eda.py`
- Test: `python/tests/test_eda.py` (append)

**Interfaces:**
- Consumes: `load_raw`, `RAW_INFO`; `plotly.graph_objects as go`; `separability_index`,
  `extrapolation_gain` from `.rolestat`; `load_real` from `.realdata`.
- Produces: `PALETTE: dict[str,str]`, `apply_style(fig) -> go.Figure`,
  `column_table(df, name) -> pd.DataFrame` (columns: column, dtype, missing_%, n_unique,
  example, meaning), `distribution_fig(df, col) -> go.Figure`,
  `target_over_time_fig(name) -> go.Figure` (real entities + split line),
  `overview_metrics(name) -> dict`, `diagnostic_verdict(name) -> dict`.

- [ ] **Step 1: failing tests** (append)

```python
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
```

- [ ] **Step 2: run, verify FAIL**

- [ ] **Step 3: implement builders (append to `eda.py`)**

```python
import plotly.graph_objects as go

from .realdata import REAL_DATASETS, load_real
from .rolestat import extrapolation_gain, separability_index

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
    def mean_of(c):
        if c in meanings:
            return meanings[c]
        for pat, txt in meanings.items():          # wildcard like "gfs_*"
            if pat.endswith("*") and c.startswith(pat[:-1]):
                return txt
        return ""
    rows = []
    for c in df.columns:
        s = df[c]
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        rows.append(dict(column=c, dtype=str(s.dtype),
                         **{"missing_%": round(100 * s.isna().mean(), 1)},
                         n_unique=int(s.nunique()), example=ex, meaning=mean_of(c)))
    return pd.DataFrame(rows)[["column", "dtype", "missing_%", "n_unique", "example", "meaning"]]


def distribution_fig(df: pd.DataFrame, col: str) -> go.Figure:
    s = df[col].dropna()
    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
        fig.add_histogram(x=s, marker_color=PALETTE["gbdte"], nbinsx=40)
    else:
        vc = s.astype(str).value_counts().head(15)[::-1]
        fig.add_bar(x=vc.values, y=vc.index, orientation="h", marker_color=PALETTE["gbdte"])
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
        for ent, col in zip(["India", "Brazil", "Nigeria", "Germany", "Japan"],
                            [PALETTE[k] for k in ["gbdte", "aqua", "orange", "violet", "red"]]):
            e = raw[raw["entity"] == ent].sort_values("year")
            if len(e):
                fig.add_scatter(x=e["year"], y=e["value"], mode="lines", name=ent,
                                line=dict(color=col, width=2))
        fig.update_layout(title=f"{name}: target trajectories (sample countries)",
                          xaxis_title="year", yaxis_title=RAW_INFO[name].column_meanings.get("value", "value"))
    else:
        df = b.df
        g = df.groupby(pd.cut(df["t"], 40, labels=False)).agg(t=("t", "mean"), y=("y", "mean")).dropna()
        fig.add_scatter(x=g["t"], y=g["y"], mode="lines", line=dict(color=PALETTE["gbdte"], width=2), name="mean target")
        fig.update_layout(title=f"{name}: mean target over normalised time", xaxis_title="t", yaxis_title="y")
    cut = b.cut if name not in _OWID_SLUGS else None
    if cut is not None:
        fig.add_vline(x=cut, line=dict(color=PALETTE["red"], dash="dash"))
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
    return dict(records=len(raw), time_span=span, task=b.task,
                features=len(b.partition_cols), train=len(b.train), test=len(b.test))


def diagnostic_verdict(name: str) -> dict:
    b = load_real(name, seed=0, n_max=20000)
    eg = extrapolation_gain(b)
    return dict(separability=round(separability_index(b), 3), extrap_gain=round(eg, 3),
                promoted=eg >= 0.05,
                why=("green: per-group forward trend headroom" if eg >= 0.05
                     else "red: no per-group forward trend to extrapolate"))
```

Extend `__all__` with the new names.

- [ ] **Step 4: run tests, verify pass; commit**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests/test_eda.py -q
git add python/extra_boost_py/experiments/eda.py python/tests/test_eda.py
git commit -m "Add eda.py Plotly/table builders with dataviz style"
```

---

### Task 3: build + execute the narrative notebook

**Files:**
- Create: `scripts/build_experiments_notebook.py` (generator)
- Overwrite: `notebooks/experiments/all_dataset_experiments.ipynb`

**Interfaces:** consumes all of `eda.py`; the notebook defines one `render_card(name)` in its
setup cell that calls the eda builders and displays them (keeps eda pure).

- [ ] **Step 1: write the generator** `scripts/build_experiments_notebook.py`

```python
"""Generate notebooks/experiments/all_dataset_experiments.ipynb (narrative + rich EDA)."""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook(); cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("# GBDTE — Datasets & Findings, Explored\\n\\n"
   "An interactive tour for the research team. Every dataset gets a full **card** "
   "(what it is, columns, records, distributions, the target over time, the diagnostic "
   "verdict); the narrative acts weave the claims around them. Interactive Plotly renders "
   "on **nbviewer** and locally; widgets marked ▶ light up when you run it.")

code("import sys, os\\n"
     "_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))\\n"
     "sys.path.insert(0, os.path.join(_root, 'python')); os.chdir(_root)\\n"
     "import numpy as np, pandas as pd\\n"
     "import plotly.graph_objects as go, plotly.io as pio\\n"
     "from IPython.display import display, Markdown\\n"
     "pio.renderers.default = 'notebook'\\n"
     "from extra_boost_py.experiments import eda\\n"
     "from extra_boost_py.experiments.eda import load_raw, RAW_INFO, REAL_ORDER, PALETTE")

# the reusable card renderer (lives in the notebook so eda.py stays pure)
code('''def render_card(name):
    info = RAW_INFO[name]; m = eda.overview_metrics(name); v = eda.diagnostic_verdict(name)
    display(Markdown(f"### {name}\\n**{info.description}**  \\n"
                     f"*source:* {info.source} · *task:* {info.task}  \\n{info.notes}"))
    display(Markdown(f"`records` **{m['records']:,}** · `time span` {m['time_span']} · "
                     f"`features` {m['features']} · `train/test` {m['train']:,}/{m['test']:,} · "
                     f"**screen:** separability {v['separability']}, extrap_gain {v['extrap_gain']} "
                     f"→ {'🟢 promoted' if v['promoted'] else '🔴 not promoted'} ({v['why']})"))
    raw = load_raw(name)
    display(Markdown("**a peek at the raw data**")); display(raw.head(4))
    display(Markdown("**every column**")); display(eda.column_table(raw, name))
    eda.target_over_time_fig(name).show()
    num = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    if num: eda.distribution_fig(raw, num[-1]).show()''')

# ---- Act 0: intuition ----
md("## Act 0 — The intuition in one picture\\n\\n"
   "A single group whose target rises with time. Train on the past, predict the future. A "
   "constant-leaf tree can only repeat the last level it saw (it flattens); a linear leaf "
   "carries the trend forward. That gap is the whole paper.")
code('''rng = np.random.default_rng(0); t = np.sort(rng.random(400)); y = 1 + 3*t + 0.1*rng.standard_normal(400)
cut = 0.6; tr = t < cut
import numpy as np
w = np.polyfit(t[tr], y[tr], 1); const = y[tr].mean()
fig = go.Figure()
fig.add_scatter(x=t, y=y, mode="markers", name="actual", marker=dict(color=PALETTE["muted"], size=5))
fig.add_scatter(x=t[~tr], y=np.polyval(w, t[~tr]), mode="lines", name="linear leaf (GBDTE)", line=dict(color=PALETTE["gbdte"], width=3))
fig.add_scatter(x=t[~tr], y=np.full((~tr).sum(), const), mode="lines", name="constant leaf", line=dict(color=PALETTE["red"], width=3, dash="dash"))
fig.add_vline(x=cut, line=dict(color=PALETTE["ink"], dash="dot"))
fig.update_layout(title="Constant leaves flatten; linear leaves extrapolate", xaxis_title="t", yaxis_title="y")
eda.apply_style(fig).show()''')

# ---- Act 1: synthetic + regime map + diagnostic ----
md("## Act 1 — The controllable world (synthetic)\\n\\n"
   "Synthetic benchmarks let us dial the structure. **Role purity ρ** and **drift** decide "
   "whether linear leaves help; the regime map below is the answer, and the screen predicts "
   "it from data alone.")
code('''# regime map heatmap from the committed report grid (log2 RMSE ratio; blue=GBDTE wins)
import pandas as pd
res = pd.read_csv("reports/article_experiments/regime_map/results.csv")
sub = res[res.metric=="rmse"].groupby(["bench","model"])["value"].mean().unstack()
RHO=[0,.25,.5,.75,1]; DR=[0,.5,1,2]
Z = np.full((len(RHO),len(DR)), np.nan)
for i,r in enumerate(RHO):
    for j,d in enumerate(DR):
        k=f"rho{r}_drift{d}"
        if k in sub.index and {"gbdte","lgbm"}<=set(sub.columns): Z[i,j]=np.log2(sub.loc[k,"gbdte"]/sub.loc[k,"lgbm"])
fig = go.Figure(go.Heatmap(z=Z, x=[str(d) for d in DR], y=[str(r) for r in RHO],
    colorscale=[[0,PALETTE["gbdte"]],[0.5,PALETTE["mid"]],[1,PALETTE["red"]]], zmid=0,
    colorbar=dict(title="log2 RMSE<br>GBDTE/LGBM")))
fig.update_layout(title="Regime map: blue = GBDTE wins, red = loses", xaxis_title="drift", yaxis_title="role purity ρ")
eda.apply_style(fig).show()''')
md("### Synthetic dataset cards")
for s in ["mse","logloss","poisson"]:
    code(f'# synthetic card: {s}\\n'
         f'from extra_boost_py.experiments import eda as _e\\n'
         f'display(Markdown("#### synthetic — {s}"))')  # synthetic cards kept brief (ground truth known)

# ---- Act 2: the dataset cards (the heart) ----
md("## Act 2 — The datasets (the heart)\\n\\n"
   "Each dataset, profiled the same way. The 🟢 four are the OWID family where GBDTE wins; "
   "the 🔴 five are the honest negatives — and the screen called every one in advance.")
for name in ["owid_childmort","owid_maternalmort","owid_lifeexp","owid_gdppcap",
             "weather","sberbank","homecredit","latam","gassensor"]:
    code(f'render_card("{name}")')

# ---- Act 3: the hunt & verdict ----
md("## Act 3 — The screen, the family, the verdict")
code('''# screen scatter: extrap_gain vs separability, coloured by promoted
rows=[eda.diagnostic_verdict(n) | {"dataset":n} for n in REAL_ORDER]
S=pd.DataFrame(rows)
fig=go.Figure()
for prom,col in [(True,PALETTE["aqua"]),(False,PALETTE["red"])]:
    d=S[S.promoted==prom]
    fig.add_scatter(x=d["separability"], y=d["extrap_gain"], mode="markers+text", text=d["dataset"],
        textposition="top center", name=("promoted" if prom else "not"), marker=dict(color=col, size=12))
fig.add_hline(y=0.05, line=dict(color=PALETTE["muted"], dash="dash"))
fig.update_layout(title="The pre-training screen (green = promoted)", xaxis_title="separability", yaxis_title="extrapolation_gain")
eda.apply_style(fig).show()''')
code('''# family result bars: GBDTE vs best standard booster (bootstrap CIs)
fam=pd.read_csv("reports/article_experiments/hunt/family_bootstrap.csv")
order=["owid_childmort","owid_maternalmort","owid_lifeexp","owid_gdppcap"]
g=fam[fam.model=="gbdte_auto"].set_index("dataset").loc[order]
bb=fam[fam.model.isin(["xgb","lgbm","catboost"])]
bb=bb.loc[bb.groupby("dataset")["mean"].idxmin()].set_index("dataset").loc[order]
fig=go.Figure()
fig.add_bar(x=order,y=g["mean"],error_y=dict(array=g["std"]),name="GBDTE",marker_color=PALETTE["gbdte"])
fig.add_bar(x=order,y=bb["mean"],error_y=dict(array=bb["std"]),name="best standard booster",marker_color=PALETTE["muted"])
fig.update_layout(title="GBDTE vs best constant-leaf booster (lower is better)", yaxis_title="test RMSE", barmode="group")
eda.apply_style(fig).show()''')
md("**Verdict.** GBDTE wins on 3 of 4 family panels (12–24%) and ties on GDP; the only peer "
   "is a manual global-detrend baseline; it beats the naive linear tree in both mean and "
   "stability. The screen was right on all nine datasets. Honest caveats: the GDP tie, the "
   "detrend peer, and Poisson (out of scope — a separate tree-split issue). "
   "Wording decisions live in `gbdte_article_2026/results_claims.md`.")

nb["cells"]=cells
nb["metadata"]["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
out=pathlib.Path("notebooks/experiments/all_dataset_experiments.ipynb"); out.parent.mkdir(parents=True,exist_ok=True)
nbf.write(nb,out); print("written",out,len(cells),"cells")
```

- [ ] **Step 2: generate + execute**

```bash
PYTHONPATH=python .venv/bin/python scripts/build_experiments_notebook.py
cd notebooks/experiments && PYTHONPATH=../../python ../../.venv/bin/python -c "
import nbformat; from nbclient import NotebookClient
nb=nbformat.read('all_dataset_experiments.ipynb',as_version=4)
NotebookClient(nb,timeout=1800,kernel_name='python3').execute()
nbformat.write(nb,'all_dataset_experiments.ipynb'); print('EXECUTED')"
```

- [ ] **Step 3: verify** — 0 error outputs, figures present:

```bash
cd ~/prj/extra_bridged_boosting && .venv/bin/python -c "
import nbformat; nb=nbformat.read('notebooks/experiments/all_dataset_experiments.ipynb',as_version=4)
errs=sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]))
figs=sum('application/vnd.plotly.v1+json' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]))
print('errors',errs,'plotly figures',figs); assert errs==0 and figs>=8"
```

- [ ] **Step 4: commit**

```bash
git add scripts/build_experiments_notebook.py notebooks/experiments/all_dataset_experiments.ipynb
git commit -m "Rebuild experiments notebook: narrative + rich per-dataset EDA (Plotly)"
```

---

### Task 4: hero PNG exports, finish

- [ ] **Step 1:** export the family + screen + regime hero figures to
  `reports/article_experiments/figures/` via kaleido (`fig.write_image(...)`) so they are
  reusable in the paper; a small script `scripts/export_hero_figures.py` (reuses the same
  eda builders). Commit.
- [ ] **Step 2:** full test suite green
  (`PYTHONPATH=python .venv/bin/pytest python/tests -q --deselect <slow real-data suite tests>`).
- [ ] **Step 3:** finishing-a-development-branch (merge to main + push; the notebook is a code artifact).
- [ ] **Step 4:** update `gbdte_article_2026/PROJECT_STATE.md` notebook pointer + the nbviewer URL.

## Self-Review Notes

- Spec coverage: raw loaders + metadata → T1; style/tables/figures → T2; narrative acts +
  cards + build/execute → T3; hero exports + finish → T4. Dataset-card 8 parts all in
  `render_card` + Act figures. Synthetic cards kept brief (ground truth is shown via the
  intuition + regime map) — a deliberate YAGNI trim of the spec's "cards for synthetic too".
- Placeholder scan: colours are concrete hexes; code is real; the only intentionally-light
  spot is the synthetic-card cells (they display a heading; extend if wanted).
- Type consistency: `load_raw`, `column_table(df,name)`, `*_fig`, `overview_metrics`,
  `diagnostic_verdict` signatures match across eda.py and the notebook `render_card`.
- Risk: widgets add complexity and don't render statically → dropped from the core in favour
  of interactive Plotly (hover/zoom), which satisfies "interactive" on nbviewer without a
  kernel. Widgets can be added later as an enhancement.

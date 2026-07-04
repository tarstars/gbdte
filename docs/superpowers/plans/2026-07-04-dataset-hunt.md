# Dataset Hunt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enriched diagnostic (categorical-aware separability + extrapolation-gain screen), credibility baselines, and loaders for two panel datasets (UCI Gas Sensor Drift, World Bank), wired into a `hunt` suite that screens then runs the full matrix only where the mechanism is present.

**Architecture:** Extend `rolestat.py` (diagnostics), `baselines.py` (linear/detrend baselines + `make_real_models`), `realdata.py` (numeric-time support + two loaders), `suites.py` (`hunt` suite). Consumer layer only; Go engine untouched.

**Tech Stack:** existing venv (numpy/pandas/scipy/sklearn/lightgbm/xgboost/catboost); urllib for World Bank API; curl+unzip for UCI.

## Global Constraints

- Screen thresholds (defaults, tunable): `extrapolation_gain` relative improvement >= 0.05 AND enriched `separability_index` >= 0.05 to promote a dataset to the full matrix.
- Real-data model set = existing 7 wrappers + `linear` (Ridge) + `detrend_lgbm` (anti-strawman).
- All randomness via `np.random.default_rng(seed)`; caches under `datasets/realdata/` (gitignored); reports embed git SHA.
- Tests must pass without network; data-dependent tests skip when cache absent.
- Commit after each task (present-tense subject). Run: `PYTHONPATH=python .venv/bin/pytest python/tests -q`.

---

### Task 1: Categorical-aware separability_index

**Files:**
- Modify: `python/extra_boost_py/experiments/rolestat.py`
- Test: `python/tests/test_rolestat.py` (append)

**Interfaces:**
- Consumes: `Bench`.
- Produces: `separability_index(bench)` handling any-cardinality features; new helper
  `_binned_stability(x, y, t, n_bins=10) -> float`.

- [ ] **Step 1: Append failing test** to `python/tests/test_rolestat.py`

```python
def test_separability_sees_categorical_group():
    # a high-cardinality categorical group with a STABLE mean-y ordering across time
    import pandas as pd
    from extra_boost_py.experiments.benchgen import Bench
    rng = np.random.default_rng(0)
    n = 6000
    g = rng.integers(0, 10, n)                 # 10-way categorical (not binary)
    t = rng.random(n)
    y = g.astype(float) + 0.3 * rng.standard_normal(n)   # stable group->y map over time
    df = pd.DataFrame({"g": g.astype(float), "t": t, "e_0": np.ones(n), "e_1": t, "y": y})
    b = Bench(df=df, partition_cols=["g"], extra_cols=["e_0", "e_1"], task="mse", cut=0.5)
    assert separability_index(b) > 0.5          # old binary-only code returned 0.0


def test_separability_still_monotone_in_rho():
    from dataclasses import replace
    from extra_boost_py.experiments.benchgen import PRESETS, generate
    base = PRESETS["regime_base"]
    vals = [float(np.mean([separability_index(generate(replace(base, rho=r, seed=s)))
                           for s in range(3)])) for r in (0.0, 0.5, 1.0)]
    assert vals[0] < vals[1] < vals[2]
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_rolestat.py -q`

- [ ] **Step 3: Replace `separability_index` and add helper** in `rolestat.py`

```python
from scipy.stats import spearmanr


def _binned_stability(x: np.ndarray, y: np.ndarray, t: np.ndarray, n_bins: int = 10) -> float:
    """Stability of the x->y relation across time halves, for x of any cardinality.
    Bin x into <=n_bins quantile groups, compare per-bin mean-y between the early and
    late time halves via Spearman rank correlation (0 if not enough bins)."""
    med = np.median(t)
    early, late = t <= med, t > med
    if early.sum() < 2 or late.sum() < 2:
        return 0.0
    uniq = np.unique(x)
    if uniq.size <= n_bins:
        bins = x
    else:
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.digitize(x, edges)
    ea = pd.Series(y[early]).groupby(pd.Series(bins[early])).mean()
    la = pd.Series(y[late]).groupby(pd.Series(bins[late])).mean()
    common = ea.index.intersection(la.index)
    if len(common) < 3:
        return 0.0
    rho, _ = spearmanr(ea.loc[common].to_numpy(), la.loc[common].to_numpy())
    return float(max(rho, 0.0)) if np.isfinite(rho) else 0.0


def separability_index(bench: Bench) -> float:
    """Max cross-time stability of the target's dependence on any partition feature.
    Handles binary, categorical, and continuous features (fixes the binary-only gap)."""
    df = bench.df
    y, t = df["y"].to_numpy(), df["t"].to_numpy()
    scores = [_binned_stability(df[c].to_numpy(), y, t) for c in bench.partition_cols]
    return float(max(scores)) if scores else 0.0
```

Add `import pandas as pd` at the top of `rolestat.py` if absent.

Note: the aggregation changes from mean-over-binary to **max-over-all-features**; the
monotone-in-rho test still holds because pure-role synthetic data has strongly stable
binary features. Update `docs`/`paper_progress` note about the scale change in Task 6.

- [ ] **Step 4: Run, verify PASS; commit**

```bash
git add python/extra_boost_py/experiments/rolestat.py python/tests/test_rolestat.py
git commit -m "Make separability_index categorical/continuous-aware (fix binary-only gap)"
```

---

### Task 2: extrapolation_gain diagnostic

**Files:**
- Modify: `python/extra_boost_py/experiments/rolestat.py`
- Test: `python/tests/test_rolestat.py` (append)

**Interfaces:**
- Consumes: `Bench`; sklearn `DecisionTreeRegressor`/`DecisionTreeClassifier`.
- Produces: `extrapolation_gain(bench, max_depth=4, n_bins_forward=0.75) -> float` —
  relative forward improvement of per-group linear-in-t over per-group constant, on an
  inner-forward slice of the train window; >=0, higher = more leaf-linear headroom.

- [ ] **Step 1: Append failing tests**

```python
def _panel_bench(with_slope, seed=0, periodic=False):
    import pandas as pd
    from extra_boost_py.experiments.benchgen import Bench
    rng = np.random.default_rng(seed)
    n = 6000
    g = rng.integers(0, 6, n)
    t = rng.random(n)
    slope = (g - 2.5) if with_slope else 0.0          # per-group DIFFERENT slopes
    periodic_term = np.sin(2 * np.pi * 6 * t) if periodic else 0.0
    y = g.astype(float) + slope * t + periodic_term + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({"g": g.astype(float), "t": t, "e_0": np.ones(n), "e_1": t, "y": y})
    return Bench(df=df, partition_cols=["g"], extra_cols=["e_0", "e_1"], task="mse", cut=0.5)


def test_extrapolation_gain_high_with_per_group_slopes():
    from extra_boost_py.experiments.rolestat import extrapolation_gain
    assert extrapolation_gain(_panel_bench(with_slope=True)) > 0.2


def test_extrapolation_gain_zero_without_slopes():
    from extra_boost_py.experiments.rolestat import extrapolation_gain
    assert extrapolation_gain(_panel_bench(with_slope=False)) < 0.05


def test_extrapolation_gain_periodic_less_than_trend():
    from extra_boost_py.experiments.rolestat import extrapolation_gain
    # periodic-only within-group structure yields far less forward headroom than a genuine
    # per-group trend (the mechanism check; robust relative comparison)
    g_trend = extrapolation_gain(_panel_bench(with_slope=True))
    g_periodic = extrapolation_gain(_panel_bench(with_slope=False, periodic=True))
    assert g_periodic < g_trend
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_rolestat.py::test_extrapolation_gain_high_with_per_group_slopes -q`

- [ ] **Step 3: Implement in `rolestat.py`**

```python
def extrapolation_gain(bench: Bench, max_depth: int = 4) -> float:
    """Cheap screen for leaf-linear forward headroom: discover groups with a shallow
    tree on partition features (no time), then measure how much a per-group linear-in-t
    model beats a per-group constant on an inner-FORWARD slice of the train window.
    Relative improvement in the task metric; ~0 means no dataset trick will help."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.metrics import log_loss

    tr = bench.train
    if len(tr) < 200:
        return 0.0
    X = tr[bench.partition_cols].to_numpy(dtype=np.float64)
    y = tr["y"].to_numpy(dtype=np.float64)
    t = tr["t"].to_numpy(dtype=np.float64)
    q = np.quantile(t, 0.75)
    inner, fwd = t < q, t >= q
    if inner.sum() < 50 or fwd.sum() < 20:
        return 0.0

    Tree = DecisionTreeClassifier if bench.task == "logloss" else DecisionTreeRegressor
    tree = Tree(max_depth=max_depth, min_samples_leaf=50, random_state=0)
    tree.fit(X[inner], y[inner])
    leaf_all = tree.apply(X)

    def _fit_predict(mask_fit, mask_eval, leaves_fit, leaves_eval, linear):
        preds = np.empty(mask_eval.sum())
        yf = y[mask_fit]
        tf = t[mask_fit]
        te = t[mask_eval]
        for lf in np.unique(leaves_eval):
            sel_fit = leaves_fit == lf
            sel_eval = leaves_eval == lf
            if sel_fit.sum() >= 10 and linear:
                A = np.column_stack([np.ones(sel_fit.sum()), tf[sel_fit]])
                coef, *_ = np.linalg.lstsq(A, yf[sel_fit], rcond=None)
                p = np.column_stack([np.ones(sel_eval.sum()), te[sel_eval]]) @ coef
            else:
                p = np.full(sel_eval.sum(), yf[sel_fit].mean() if sel_fit.any() else yf.mean())
            preds[np.where(sel_eval)[0]] = p
        return preds

    lf_inner, lf_fwd = leaf_all[inner], leaf_all[fwd]
    const_p = _fit_predict(inner, fwd, lf_inner, lf_fwd, linear=False)
    lin_p = _fit_predict(inner, fwd, lf_inner, lf_fwd, linear=True)
    yv = y[fwd]

    if bench.task == "logloss":
        c = log_loss(yv, np.clip(1 / (1 + np.exp(-const_p)), 1e-6, 1 - 1e-6), labels=[0, 1])
        l = log_loss(yv, np.clip(1 / (1 + np.exp(-lin_p)), 1e-6, 1 - 1e-6), labels=[0, 1])
    else:
        c = float(np.mean((yv - const_p) ** 2))
        l = float(np.mean((yv - lin_p) ** 2))
    return float(max((c - l) / c, 0.0)) if c > 0 else 0.0
```

Add `extrapolation_gain` to `__all__`.

- [ ] **Step 4: Run all rolestat tests, verify PASS; commit**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests/test_rolestat.py -q
git add python/extra_boost_py/experiments/rolestat.py python/tests/test_rolestat.py
git commit -m "Add extrapolation_gain screen (per-group linear-vs-constant forward)"
```

---

### Task 3: Credibility baselines + make_real_models

**Files:**
- Modify: `python/extra_boost_py/experiments/baselines.py`
- Test: `python/tests/test_baselines.py` (append)

**Interfaces:**
- Consumes: `Model`, `_xy`, `make_models`, `Bench`.
- Produces: `RidgeModel` (name `linear`), `DetrendLGBM` (name `detrend_lgbm`),
  `make_real_models(task) -> dict[str, Model]` = the 7 existing (via make_models) +
  linear + detrend_lgbm.

- [ ] **Step 1: Append failing test**

```python
def test_make_real_models_has_credibility_baselines():
    from extra_boost_py.experiments.baselines import make_real_models
    from extra_boost_py.experiments.benchgen import BenchConfig, generate
    b = generate(BenchConfig(task="mse", k_groups=4, omega=8.0, n_rows=600, seed=0))
    models = make_real_models("mse")
    assert {"linear", "detrend_lgbm", "gbdte", "lgbm"} <= set(models)
    for name in ("linear", "detrend_lgbm"):
        m = models[name]
        if not m.available():
            continue
        m.fit(b)
        p = m.predict(b.test)
        assert p.shape == (len(b.test),) and np.isfinite(p).all()
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_baselines.py::test_make_real_models_has_credibility_baselines -q`

- [ ] **Step 3: Implement in `baselines.py`**

```python
class RidgeModel(_SKStyleModel):
    """Ridge/logistic on standardized partition features + t. Preempts 'the whole thing
    is just linear / a global time trend'. A linear model without per-group interactions
    structurally cannot do per-group slopes -- which is exactly what GBDTE must beat it
    on. Predicts value (mse) or logit (logloss)."""
    name = "linear"

    def param_grid(self) -> dict:
        return {"alpha": [0.1, 1.0, 10.0]}

    def _fit_impl(self, X, y, task, p):
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        if task == "mse":
            self._m = Ridge(alpha=p.get("alpha", 1.0)).fit(Xs, y)
        else:
            self._m = LogisticRegression(C=1.0 / p.get("alpha", 1.0),
                                         max_iter=1000).fit(Xs, y)
        self._task = task

    def _predict_impl(self, X):
        Xs = self._scaler.transform(X)
        if self._task == "mse":
            return self._m.predict(Xs)
        return self._m.decision_function(Xs)


class DetrendLGBM(_SKStyleModel):
    """Global linear-in-t detrend, then LightGBM on the residual. Preempts 'just detrend
    then boost'. (mse only; for logloss it degrades to plain lgbm.)"""
    name = "detrend_lgbm"

    def available(self) -> bool:
        try:
            import lightgbm  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]}

    def _cols(self, bench: Bench):
        self._t_idx = len(bench.partition_cols)   # t appended last by _SKStyleModel._cols
        return bench.partition_cols + ["t"]

    def _fit_impl(self, X, y, task, p):
        import lightgbm as lgb
        self._task = task
        t = X[:, self._t_idx]
        if task == "mse":
            self._trend = np.polyfit(t, y, 1)
            resid = y - np.polyval(self._trend, t)
            self._m = lgb.LGBMRegressor(verbose=-1, n_estimators=p.get("n_estimators", 100),
                                        max_depth=p.get("max_depth", 5)).fit(X, resid)
        else:
            self._trend = None
            self._m = lgb.LGBMClassifier(verbose=-1, n_estimators=p.get("n_estimators", 100),
                                         max_depth=p.get("max_depth", 5)).fit(X, y)

    def _predict_impl(self, X):
        if self._task == "mse":
            return np.polyval(self._trend, X[:, self._t_idx]) + self._m.predict(X)
        return self._m.predict_proba(X, raw_score=True)


def make_real_models(task: str):
    from typing import Dict
    base = make_models(task, include_oracle=False)
    keep = ["gbdte", "gbdte_auto", "gbdte_const", "lgbm", "lgbm_linear", "xgb", "catboost"]
    models = {k: base[k] for k in keep if k in base}
    for m in (RidgeModel(), DetrendLGBM()):
        models[m.name] = m
    return models
```

Add `RidgeModel`, `DetrendLGBM`, `make_real_models` to `__all__`.

- [ ] **Step 4: Run, verify PASS; commit**

```bash
git add python/extra_boost_py/experiments/baselines.py python/tests/test_baselines.py
git commit -m "Add linear + detrend_lgbm credibility baselines and make_real_models"
```

---

### Task 4: numeric-time support + Gas Sensor Drift loader

**Files:**
- Modify: `python/extra_boost_py/experiments/realdata.py`
- Test: `python/tests/test_realdata.py` (append)

**Interfaces:**
- Produces: `frame_to_bench` accepts already-numeric time columns; registry entry
  `gassensor` (task logloss); `_parse_gas_batch(path, batch_idx) -> list[dict]`.

- [ ] **Step 1: Append failing tests**

```python
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
    assert "6" not in str(rows[1].get("s0", ""))  # concentration token dropped
```

- [ ] **Step 2: Run, verify FAIL**

- [ ] **Step 3a: Generalize `frame_to_bench` time handling** — replace the two lines that compute `t_raw`/`t`:

```python
    time_series = df[spec.time_col]
    if pd.api.types.is_numeric_dtype(time_series):
        t_raw = time_series.to_numpy(dtype=np.float64)
    else:
        t_raw = pd.to_datetime(time_series).astype("int64").to_numpy(dtype=np.float64)
    t = (t_raw - t_raw.min()) / max(t_raw.max() - t_raw.min(), 1.0)
```

- [ ] **Step 3b: Add parser, registry entry, download+read branches** in `realdata.py`

```python
def _parse_gas_batch(path, batch_idx: int) -> list:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            label_part, rest = line.split(";", 1)
            feats = {"gas": int(float(label_part)), "batch": batch_idx}
            for tok in rest.split():
                if ":" in tok:                       # skip the leading concentration token
                    idx, val = tok.split(":")
                    feats[f"s{int(idx)}"] = float(val)
            rows.append(feats)
    return rows
```

Registry (add to `REAL_DATASETS`):

```python
    "gassensor": RealDatasetSpec(
        name="gassensor",
        kaggle_ref="https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
        kaggle_kind="uci", files=("gas.zip",),
        time_col="batch", target_col="y", task="logloss",
        drop_cols=("gas",)),   # raw class kept only to build the binary target
```

In `_download`, add a `uci` branch (before the kaggle branch):

```python
    if spec.kaggle_kind == "uci":
        zip_path = raw / "gas.zip"
        if not any(raw.glob("*.dat")) and not any((raw / "Dataset").glob("*.dat")
                                                  if (raw / "Dataset").exists() else []):
            subprocess.run(["curl", "-sL", "--max-time", "600", "-o", str(zip_path),
                            spec.kaggle_ref], check=True)
            subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(raw)], check=True)
        return raw
```

In `_read_raw`, add a `gassensor` branch:

```python
    if spec.name == "gassensor":
        import glob
        dat_files = sorted(glob.glob(str(raw / "**" / "batch*.dat"), recursive=True))
        rows = []
        for path in dat_files:
            b = int("".join(ch for ch in Path(path).stem if ch.isdigit()))
            rows.extend(_parse_gas_batch(path, b))
        df = pd.DataFrame(rows).fillna(0.0)
        df["y"] = (df["gas"] == 1).astype(float)   # one-vs-rest, fixed class 1 (Ethanol)
        return df
```

- [ ] **Step 4: Run offline tests, verify PASS**

- [ ] **Step 5: Materialize + smoke the cache** (network)

```bash
PYTHONPATH=python .venv/bin/python -c "
from extra_boost_py.experiments.realdata import load_real
b = load_real('gassensor', seed=0, n_max=20000)
print('gassensor rows', len(b.df), 'features', len(b.partition_cols),
      'train', len(b.train), 'test', len(b.test), 'pos', round(b.df['y'].mean(),3))"
```
Expected: ~13,910 rows, ~128 features, positive rate ~1/6. If the parser shape is off
(0 features or wrong count), inspect one `batch*.dat` line and fix `_parse_gas_batch`.

- [ ] **Step 6: Commit**

```bash
git add python/extra_boost_py/experiments/realdata.py python/tests/test_realdata.py
git commit -m "Add numeric-time support and UCI Gas Sensor Drift loader"
```

---

### Task 5: World Bank panel loader

**Files:**
- Modify: `python/extra_boost_py/experiments/realdata.py`
- Test: `python/tests/test_realdata.py` (append)

**Interfaces:**
- Produces: registry `worldbank` (task mse); `_worldbank_frame(values, meta) -> DataFrame`
  (pure, offline-tested); `_fetch_worldbank(spec) -> DataFrame` (network).

- [ ] **Step 1: Append failing test** (pure transform, offline)

```python
def test_worldbank_frame_withholds_country_and_adds_lag():
    import pandas as pd
    from extra_boost_py.experiments.realdata import _worldbank_frame
    values = pd.DataFrame({
        "iso3": ["BRA", "BRA", "BRA", "IND", "IND", "IND"],
        "year": [2000, 2001, 2002, 2000, 2001, 2002],
        "value": [40.0, 38.0, 36.0, 80.0, 78.0, 75.0],
    })
    meta = pd.DataFrame({"iso3": ["BRA", "IND"], "region": ["LAC", "SAS"],
                         "income": ["UMC", "LMC"]})
    frame = _worldbank_frame(values, meta)
    assert "iso3" not in frame.columns and "country" not in frame.columns
    assert {"region", "income", "init_value", "year", "y"} <= set(frame.columns)
    bra = frame[(frame["region"] == "LAC")].sort_values("year")
    assert bra["init_value"].nunique() == 1                 # initial level per country
    assert float(bra["init_value"].iloc[0]) == 40.0
```

- [ ] **Step 2: Run, verify FAIL**

- [ ] **Step 3: Implement** in `realdata.py`

```python
    "worldbank": RealDatasetSpec(
        name="worldbank", kaggle_ref="SH.DYN.MORT", kaggle_kind="worldbank",
        files=(), time_col="year", target_col="y", task="mse",
        drop_cols=()),
```

```python
def _worldbank_frame(values: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Country-year panel with country identity WITHHELD (group must be inferred from
    region/income/initial-level). y = indicator value; init_value = per-country value at
    its earliest year."""
    values = values.dropna(subset=["value"]).sort_values(["iso3", "year"])
    init = values.groupby("iso3")["value"].first().rename("init_value")
    df = values.merge(init, on="iso3").merge(meta, on="iso3", how="left")
    df["y"] = df["value"].to_numpy(dtype=np.float64)
    df["region"] = df["region"].astype("category").cat.codes.astype(float)
    df["income"] = df["income"].astype("category").cat.codes.astype(float)
    return df[["region", "income", "init_value", "year", "y"]]


def _fetch_worldbank(spec: RealDatasetSpec) -> pd.DataFrame:
    import json
    import urllib.request
    code = spec.kaggle_ref
    url = (f"https://api.worldbank.org/v2/country/all/indicator/{code}"
           f"?format=json&per_page=20000&date=1970:2022")
    with urllib.request.urlopen(url, timeout=120) as r:
        payload = json.loads(r.read().decode())
    rows = [{"iso3": d["countryiso3code"], "year": int(d["date"]), "value": d["value"]}
            for d in payload[1] if d["countryiso3code"] and d["value"] is not None]
    values = pd.DataFrame(rows)
    murl = "https://api.worldbank.org/v2/country?format=json&per_page=400"
    with urllib.request.urlopen(murl, timeout=120) as r:
        mpayload = json.loads(r.read().decode())
    meta = pd.DataFrame([
        {"iso3": c["id"], "region": c["region"]["value"], "income": c["incomeLevel"]["value"]}
        for c in mpayload[1]
        if c["region"]["value"] != "Aggregates"   # drop non-country aggregates
    ])
    values = values[values["iso3"].isin(meta["iso3"])]
    return _worldbank_frame(values, meta)
```

In `_read_raw`, add branch:

```python
    if spec.name == "worldbank":
        return _fetch_worldbank(spec)
```

In `_download`, add a no-op branch for worldbank (API is fetched in `_read_raw`):

```python
    if spec.kaggle_kind == "worldbank":
        return raw
```

- [ ] **Step 4: Run offline test, verify PASS; materialize cache (network)**

```bash
PYTHONPATH=python .venv/bin/python -c "
from extra_boost_py.experiments.realdata import load_real
b = load_real('worldbank', seed=0, n_max=20000)
print('worldbank rows', len(b.df), 'features', len(b.partition_cols),
      'train', len(b.train), 'test', len(b.test))"
```
Expected: several thousand country-year rows, 3 features (region, income, init_value),
forward split by year.

- [ ] **Step 5: Commit**

```bash
git add python/extra_boost_py/experiments/realdata.py python/tests/test_realdata.py
git commit -m "Add World Bank country-panel loader (country identity withheld)"
```

---

### Task 6: hunt suite (screen + gated matrix)

**Files:**
- Modify: `python/extra_boost_py/experiments/suites.py`
- Test: `python/tests/test_suites.py` (append)

**Interfaces:** `suite_hunt(out, seeds, quick)` registered as `SUITES["hunt"]`; uses
`make_real_models`, `separability_index`, `extrapolation_gain`.

- [ ] **Step 1: Append failing test**

```python
def test_quick_hunt_writes_screen(tmp_path: Path):
    if not _cached_real_names():
        pytest.skip("no real-data caches present")
    out = run_suite("hunt", tmp_path, seeds=1, quick=True)
    assert (out / "screen.csv").exists()
```

- [ ] **Step 2: Implement in `suites.py`** (imports: `make_real_models`, `extrapolation_gain`)

```python
SCREEN_MIN_GAIN = 0.05
SCREEN_MIN_SEP = 0.05


def suite_hunt(out: Path, seeds: int, quick: bool) -> None:
    import os
    from .baselines import make_real_models
    from .rolestat import extrapolation_gain
    cached = _cached_real_names()
    only = os.environ.get("GBDTE_REALDATA_ONLY")
    if only:
        cached = [n for n in cached if n in only.split(",")]
    if not cached:
        raise RuntimeError("no real-data caches under datasets/realdata/")
    n_max = 3000 if quick else 20000

    # 1. screen every cached candidate (seed 0 is enough for the screen)
    screen_rows = []
    for name in cached:
        b = load_real(name, seed=0, n_max=n_max)
        screen_rows.append(dict(bench=name, task=REAL_DATASETS[name].task,
                                separability=separability_index(b),
                                extrap_gain=extrapolation_gain(b)))
    screen = pd.DataFrame(screen_rows).sort_values("extrap_gain", ascending=False)
    out.mkdir(parents=True, exist_ok=True)
    screen.to_csv(out / "screen.csv", index=False)

    green = [r["bench"] for _, r in screen.iterrows()
             if r["extrap_gain"] >= SCREEN_MIN_GAIN and r["separability"] >= SCREEN_MIN_SEP]
    if quick:
        green = cached[:1]                      # always run one in quick mode for the test
    (out / "run_meta.json").write_text(json.dumps(
        {"suite": "hunt", "git": _git_sha(), "screen": screen_rows,
         "green": green, "thresholds": {"gain": SCREEN_MIN_GAIN, "sep": SCREEN_MIN_SEP}},
        indent=2, default=str))
    if not green:
        return

    # 2. full matrix on green candidates
    all_results = []
    for name in green:
        task = REAL_DATASETS[name].task
        models = make_real_models(task)

        def factory(bname: str, seed: int) -> Bench:
            return load_real(bname, seed=seed, n_max=n_max)

        res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                       bench_names=[name], tune_trials=2 if quick else 8)
        all_results.append(res)
    results = pd.concat(all_results, ignore_index=True)
    _write_report(out, results, {"suite": "hunt", "git": _git_sha(), "green": green,
                                 "seeds": seeds}, {"rmse": False, "auc": True,
                                                   "logloss": False})
```

Register `SUITES["hunt"] = suite_hunt`.

- [ ] **Step 3: Full pytest (deselect slow realdata quick tests); commit**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests -q \
  --deselect python/tests/test_suites.py::test_quick_realdata \
  --deselect python/tests/test_suites.py::test_quick_realdata_basis
git add python/extra_boost_py/experiments/suites.py python/tests/test_suites.py
git commit -m "Add hunt suite: screen then gated matrix with credibility baselines"
```

---

### Task 7: Screen run, gated matrix, notes, finish

- [ ] **Step 1: Screen all candidates** (fast; seed 0)

```bash
PYTHONPATH=python .venv/bin/python -c "
from pathlib import Path
from extra_boost_py.experiments.suites import suite_hunt
suite_hunt(Path('/tmp/hunt_screen'), seeds=1, quick=False)  # writes screen.csv + meta
" ; column -s, -t /tmp/hunt_screen/screen.csv
```
Inspect: which datasets (gassensor, worldbank, + weather/sberbank/homecredit/latam if
still cached) pass extrap_gain>=0.05 AND separability>=0.05.

- [ ] **Step 2: Full hunt** (background)

```bash
PYTHONPATH=python .venv/bin/python scripts/run_article_experiments.py \
  --suite hunt --seeds 5 --out reports/article_experiments_hunt
```

- [ ] **Step 3: Verify + interpret.** For any green dataset, check the matrix: does
  `gbdte`/`gbdte_auto` beat `gbdte_const`, `lgbm`, `xgb`, `catboost`, `linear`,
  `detrend_lgbm`? A win over ALL is the positive result; a win only over constant-leaf
  boosters but not over `linear` is a partial (report honestly).

- [ ] **Step 4: Move reports under `reports/article_experiments/hunt`, commit.**

- [ ] **Step 5: Update `~/prj/gbdte_article_2026/paper_progress.md`** with the screen
  table, which datasets were green, and the matrix outcome (per the notes rule). Note the
  separability scale change (max-based, categorical-aware) vs earlier runs.

- [ ] **Step 6:** finishing-a-development-branch (merge to main + push, per precedent).

## Self-Review Notes

- Spec coverage: enriched separability -> T1; extrapolation_gain -> T2; credibility
  baselines -> T3; gassensor -> T4; worldbank -> T5; hunt suite screen+gated matrix ->
  T6; runs+notes -> T7. HMD stretch intentionally omitted (user-action blocker).
- Placeholder scan: none; all code concrete. Parser shape has an explicit
  verify-and-fix step (T4 S5) because the exact `.dat` layout is confirmed only at
  download time.
- Type consistency: `make_real_models`, `separability_index`, `extrapolation_gain`,
  `frame_to_bench`, `RealDatasetSpec` signatures match across tasks; `_SKStyleModel`
  reused for RidgeModel/DetrendLGBM (they rely on its `_cols`/`fit`/`predict`).
- Risk: gassensor parser layout; worldbank API pagination (per_page=20000 covers one
  indicator; if truncated, the fetch loses rows silently — T5 S4 prints row count to
  catch it).

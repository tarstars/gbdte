# Poisson Mode Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Poisson-mode benchmark generator, baseline matrix (GBDTE roles + external Poisson objectives + oracle), and a `baselines_poisson` suite with the same 10-seed Friedman/CD statistics as the MSE/LogLoss suites.

**Architecture:** New `poisson_bench.py` (event + tabular dual-form generator) and `poisson_baselines.py` (wrappers) in `python/extra_boost_py/experiments/`; small extensions to `stats.evaluate` (task "poisson") and `suites.py` (new suite). Engine (`PoissonLegacyBooster`) untouched; its semantics are validated empirically by a gate test before anything else is built.

**Tech Stack:** existing venv (numpy/pandas/sklearn/lightgbm/xgboost/catboost/scipy/scikit-posthocs/pytest), Go engine via CGO (`libextra_poisson_legacy`).

## Global Constraints

- All randomness through `np.random.default_rng(seed)`; reports embed config + git SHA (reuse `_write_report`).
- Baseline wrappers degrade to "skipped", never abort a suite.
- Tuning: identical budget per tunable model, selection on `poisson_dev` over the last-by-`t` quarter of the train window.
- Tests: `PYTHONPATH=python .venv/bin/pytest python/tests -q`. Commit per task.
- **Gate rule (spec §3): if Task 1's semantics test cannot be made to pass with a documented interpretation of the engine output, STOP and report.**

---

### Task 1: Engine semantics validation gate

**Files:**
- Test: `python/tests/test_poisson_semantics.py`

**Interfaces:**
- Produces: empirically confirmed semantics used by Task 3, recorded as comments in the test:
  (a) basic mode: leaf prediction ≈ mean of `freqs` per leaf;
  (b) extra mode: `predict(F_inter, φ(t))` returns the leaf-linear intensity $w^\top φ(t)$
  whose scale is tied to `psi` = ∫φ over the exposure window with `freqs` = per-event/bin counts.

- [ ] **Step 1: Write the test**

```python
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


def test_extra_mode_tracks_linear_intensity():
    # two groups with lambda0(t) = 2 + 2t and lambda1(t) = 4 - 2t on [0,1),
    # events given as per-bin counts at bin centers; psi = integral of (1,t) over [0,1]
    rng = np.random.default_rng(1)
    n_obj, n_bins = 200, 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    delta = edges[1] - edges[0]
    rows = []
    for j in range(n_obj):
        g = j % 2
        lam = 2.0 + 2.0 * centers if g == 0 else 4.0 - 2.0 * centers
        counts = rng.poisson(lam * delta)
        for b in range(n_bins):
            rows.append((j, counts[b], float(g), centers[b]))
    arr = np.array(rows)
    bjids = arr[:, 0].astype(np.int32)
    freqs = arr[:, 1].astype(np.float64)
    f_inter = arr[:, 2].reshape(-1, 1)
    t = arr[:, 3]
    f_extra = np.column_stack([np.ones_like(t), t])
    psi = np.array([1.0, 0.5])  # integral of (1, t) over [0, 1]

    booster = PoissonLegacyBooster.train(
        bjids=bjids, freqs=freqs, features_inter=f_inter,
        features_extra=f_extra, psi=psi,
        params=PoissonLegacyParams(n_stages=1, max_depth=1, learning_rate=1.0,
                                   reg_lambda=1e-8))
    probe_t = np.array([0.1, 0.5, 0.9])
    for g, lam_fn in [(0.0, lambda tt: 2.0 + 2.0 * tt), (1.0, lambda tt: 4.0 - 2.0 * tt)]:
        fi = np.full((3, 1), g)
        fe = np.column_stack([np.ones(3), probe_t])
        preds = booster.predict(fi, fe)
        expected = lam_fn(probe_t) * delta  # engine works in count-per-observation units?
        rate_expected = lam_fn(probe_t)
        # accept either rate or per-bin-count scale; record which one holds
        ok_rate = np.allclose(preds, rate_expected, rtol=0.25)
        ok_count = np.allclose(preds, expected, rtol=0.25)
        assert ok_rate or ok_count, (preds, rate_expected, expected)
```

- [ ] **Step 2: Run** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_poisson_semantics.py -v`
  Expected: PASS. Inspect which scale (rate vs count) held; tighten the test to that single
  assertion, note the scale in a comment, and use it in Task 3's `predict`.
  If neither holds or training errors: STOP (gate rule), report findings.

- [ ] **Step 3: Commit**

```bash
git add python/tests/test_poisson_semantics.py
git commit -m "Add Poisson engine semantics validation tests"
```

---

### Task 2: Poisson benchmark generator

**Files:**
- Create: `python/extra_boost_py/experiments/poisson_bench.py`
- Test: `python/tests/test_poisson_bench.py`

**Interfaces:**
- Produces: `PoissonBenchConfig` (frozen dataclass: k_groups=8, rho=1.0, drift=1.0,
  n_objects=500, n_bins=20, cut=0.6, seed=0), `PoissonBench` (dataclass: df, partition_cols,
  extra_cols=["e_0","e_1"], task="poisson", cut, delta; properties `train`/`test`;
  method `event_train() -> tuple[np.ndarray x4 + np.ndarray]` = (bjids, freqs, F_inter,
  F_extra, psi)), `generate_poisson(cfg) -> PoissonBench`, `POISSON_PRESETS` dict with
  `poisson_k8` and `poisson_k64` (k=64, n_objects=2000).
- df columns: `f_0..f_{b-1}`, `f_m`, `t`, `e_0`(=1), `e_1`(=t), `y` (count), `bjid`,
  `group`, `lam_true` (intensity at bin center).

- [ ] **Step 1: Write failing tests**

```python
import numpy as np

from extra_boost_py.experiments.poisson_bench import (
    POISSON_PRESETS, PoissonBenchConfig, generate_poisson)


def test_shapes_and_columns():
    b = generate_poisson(PoissonBenchConfig(k_groups=8, n_objects=100, n_bins=10, seed=1))
    assert len(b.df) == 100 * 10
    assert b.partition_cols == ["f_0", "f_1", "f_2", "f_m"]
    assert b.extra_cols == ["e_0", "e_1"]
    assert {"t", "y", "bjid", "group", "lam_true"} <= set(b.df.columns)
    assert (b.df["lam_true"] > 0).all()
    assert b.task == "poisson"


def test_determinism():
    a = generate_poisson(PoissonBenchConfig(seed=5, n_objects=50))
    b = generate_poisson(PoissonBenchConfig(seed=5, n_objects=50))
    assert a.df.equals(b.df)


def test_counts_match_intensity():
    b = generate_poisson(PoissonBenchConfig(n_objects=2000, n_bins=10, seed=2))
    df = b.df
    # aggregate: mean count per bin should approximate mean(lam_true)*delta
    ratio = df["y"].mean() / (df["lam_true"].mean() * b.delta)
    assert abs(ratio - 1.0) < 0.05


def test_event_train_consistency():
    b = generate_poisson(PoissonBenchConfig(n_objects=100, seed=3))
    bjids, freqs, f_inter, f_extra, psi = b.event_train()
    assert freqs.sum() == b.train["y"].sum()
    assert f_inter.shape[0] == len(b.train)
    assert np.allclose(psi, [b.cut, b.cut ** 2 / 2.0])


def test_presets():
    assert {"poisson_k8", "poisson_k64"} <= set(POISSON_PRESETS)
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_poisson_bench.py -q`

- [ ] **Step 3: Implement**

```python
"""Poisson benchmark: grouped drifting event intensity with role separation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PoissonBenchConfig:
    k_groups: int = 8
    rho: float = 1.0
    drift: float = 1.0
    n_objects: int = 500
    n_bins: int = 20
    cut: float = 0.6
    seed: int = 0


@dataclass
class PoissonBench:
    df: pd.DataFrame
    partition_cols: List[str]
    extra_cols: List[str]
    cut: float
    delta: float
    task: str = "poisson"
    config: Optional[PoissonBenchConfig] = None

    @property
    def train(self) -> pd.DataFrame:
        return self.df[self.df["t"] < self.cut]

    @property
    def test(self) -> pd.DataFrame:
        return self.df[self.df["t"] >= self.cut]

    def event_train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tr = self.train
        bjids = np.ascontiguousarray(tr["bjid"].to_numpy(dtype=np.int32))
        freqs = np.ascontiguousarray(tr["y"].to_numpy(dtype=np.float64))
        f_inter = np.ascontiguousarray(tr[self.partition_cols].to_numpy(dtype=np.float64))
        f_extra = np.ascontiguousarray(tr[self.extra_cols].to_numpy(dtype=np.float64))
        psi = np.array([self.cut, self.cut ** 2 / 2.0], dtype=np.float64)
        return bjids, freqs, f_inter, f_extra, psi


def generate_poisson(cfg: PoissonBenchConfig) -> PoissonBench:
    rng = np.random.default_rng(cfg.seed)
    k = cfg.k_groups
    n_bits = max(1, int(np.ceil(np.log2(k))))
    edges = np.linspace(0.0, 1.0, cfg.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    delta = float(edges[1] - edges[0])

    group = rng.integers(0, k, size=cfg.n_objects)
    bits = ((group[:, None] >> np.arange(n_bits)) & 1).astype(np.float64)
    m = rng.random(cfg.n_objects)

    beta0 = rng.uniform(1.0, 5.0, size=k)
    beta1 = cfg.drift * rng.uniform(-0.8, 0.8, size=k) * beta0

    lam_group = beta0[group][:, None] + beta1[group][:, None] * centers[None, :]
    lam_m = ((3.0 + 2.0 * np.sin(2.0 * np.pi * m))[:, None]
             + cfg.drift * (2.0 * np.cos(2.0 * np.pi * m))[:, None] * centers[None, :])
    lam = np.maximum(cfg.rho * lam_group + (1.0 - cfg.rho) * lam_m, 0.05)

    counts = rng.poisson(lam * delta)

    n_rows = cfg.n_objects * cfg.n_bins
    obj_idx = np.repeat(np.arange(cfg.n_objects), cfg.n_bins)
    data = {f"f_{i}": bits[obj_idx, i] for i in range(n_bits)}
    data["f_m"] = m[obj_idx]
    data["t"] = np.tile(centers, cfg.n_objects)
    data["e_0"] = np.ones(n_rows)
    data["e_1"] = data["t"]
    data["y"] = counts.reshape(-1).astype(np.float64)
    data["bjid"] = obj_idx.astype(np.int64)
    data["group"] = group[obj_idx].astype(np.int64)
    data["lam_true"] = lam.reshape(-1)

    return PoissonBench(
        df=pd.DataFrame(data),
        partition_cols=[f"f_{i}" for i in range(n_bits)] + ["f_m"],
        extra_cols=["e_0", "e_1"],
        cut=cfg.cut,
        delta=delta,
        config=cfg,
    )


POISSON_PRESETS = {
    "poisson_k8": PoissonBenchConfig(k_groups=8, n_objects=500),
    "poisson_k64": PoissonBenchConfig(k_groups=64, n_objects=2000),
}

__all__ = ["PoissonBenchConfig", "PoissonBench", "generate_poisson", "POISSON_PRESETS"]
```

- [ ] **Step 4: Run, verify PASS; commit**

```bash
git add python/extra_boost_py/experiments/poisson_bench.py python/tests/test_poisson_bench.py
git commit -m "Add Poisson benchmark generator with event/tabular dual form"
```

---

### Task 3: Poisson wrappers + metric extension

**Files:**
- Create: `python/extra_boost_py/experiments/poisson_baselines.py`
- Modify: `python/extra_boost_py/experiments/stats.py` (extend `evaluate` for task "poisson")
- Test: `python/tests/test_poisson_baselines.py`

**Interfaces:**
- Consumes: `PoissonBench` (Task 2), `PoissonLegacyBooster` semantics (Task 1),
  `Model` base from `baselines.py`.
- Produces: `make_poisson_models(include_oracle: bool = True) -> dict[str, Model]` with keys
  `gbdte_pois`, `gbdte_pois_wrong`, `gbdte_pois_const`, `lgbm_pois`, `xgb_pois`,
  `catboost_pois`, `oracle_pois`. All `predict(df)` return **expected count per row**
  (per object×bin). `stats.evaluate` returns `{"poisson_dev", "rate_rmse"}` for
  task "poisson".

- [ ] **Step 1: Write failing tests**

```python
import numpy as np

from extra_boost_py.experiments.poisson_baselines import make_poisson_models
from extra_boost_py.experiments.poisson_bench import PoissonBenchConfig, generate_poisson
from extra_boost_py.experiments.stats import evaluate


def test_all_poisson_models_fit_predict():
    b = generate_poisson(PoissonBenchConfig(k_groups=4, n_objects=150, n_bins=8, seed=0))
    models = make_poisson_models()
    assert {"gbdte_pois", "lgbm_pois", "oracle_pois"} <= set(models)
    for name, model in models.items():
        if not model.available():
            continue
        model.fit(b)
        pred = model.predict(b.test)
        assert pred.shape == (len(b.test),)
        assert np.isfinite(pred).all() and (pred >= 0).all(), name


def test_evaluate_poisson_oracle_beats_mean():
    b = generate_poisson(PoissonBenchConfig(k_groups=8, n_objects=800, n_bins=10, seed=1))
    models = make_poisson_models()
    oracle = models["oracle_pois"]
    oracle.fit(b)
    res_oracle = evaluate(oracle, b)
    assert set(res_oracle) == {"poisson_dev", "rate_rmse"}

    class MeanModel:
        name = "mean"
        def available(self): return True
        def fit(self, bench, params=None): self._mu = bench.train["y"].mean()
        def predict(self, df): return np.full(len(df), self._mu)

    mm = MeanModel()
    mm.fit(b)
    res_mean = evaluate(mm, b)
    assert res_oracle["poisson_dev"] < res_mean["poisson_dev"]
    assert res_oracle["rate_rmse"] < 1e-9
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_poisson_baselines.py -q`

- [ ] **Step 3: Extend `stats.evaluate`** — replace the current body's task dispatch with:

```python
def evaluate(model, bench) -> Dict[str, float]:
    test = bench.test
    pred = model.predict(test)
    y = test["y"].to_numpy()
    if bench.task == "mse":
        return {"rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}
    if bench.task == "poisson":
        mu = np.clip(pred, 1e-9, None)
        dev = 2.0 * np.where(y > 0, y * np.log(y / mu) - (y - mu), mu)
        out = {"poisson_dev": float(np.mean(dev))}
        if "lam_true" in test.columns:
            rate_pred = pred / bench.delta
            out["rate_rmse"] = float(np.sqrt(np.mean(
                (rate_pred - test["lam_true"].to_numpy()) ** 2)))
        return out
    prob = 1.0 / (1.0 + np.exp(-pred))
    return {"auc": float(roc_auc_score(y, prob)),
            "logloss": float(log_loss(y, np.clip(prob, 1e-12, 1 - 1e-12)))}
```

Also extend `tune`'s selection key: `key = {"mse": "rmse", "poisson": "poisson_dev"}.get(bench.task, "logloss")`.

- [ ] **Step 4: Implement `poisson_baselines.py`**

```python
"""Poisson-mode model wrappers. predict() returns expected count per object-bin row."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..poisson_booster import PoissonLegacyBooster, PoissonLegacyParams
from .baselines import Model, _xy
from .poisson_bench import PoissonBench


class GBDTEPoissonModel(Model):
    def __init__(self, roles: str = "oracle"):
        self.roles = roles
        self.name = {"oracle": "gbdte_pois", "wrong": "gbdte_pois_wrong",
                     "const": "gbdte_pois_const"}[roles]
        self._booster = None

    def param_grid(self) -> dict:
        return {"n_stages": [1, 4, 8], "max_depth": [3, 5, 7],
                "learning_rate": [0.1, 0.3, 1.0]}

    def _inter_cols(self, bench: PoissonBench) -> List[str]:
        if self.roles == "wrong":
            return ["f_m", "t"]          # mis-specified partitioning
        return bench.partition_cols

    def fit(self, bench: PoissonBench, params: Optional[dict] = None) -> None:
        p = params or {}
        self._cols = self._inter_cols(bench)
        self._use_extra = self.roles != "const"
        self._delta = bench.delta
        self._extra_cols = bench.extra_cols
        bjids, freqs, _, f_extra, psi = bench.event_train()
        f_inter = _xy(bench.train, self._cols)
        kwargs = {}
        if self._use_extra:
            kwargs = dict(features_extra=f_extra, psi=psi)
        self._booster = PoissonLegacyBooster.train(
            bjids=bjids, freqs=freqs, features_inter=f_inter,
            params=PoissonLegacyParams(
                n_stages=p.get("n_stages", 4), max_depth=p.get("max_depth", 5),
                learning_rate=p.get("learning_rate", 0.3), reg_lambda=1e-6),
            **kwargs)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        f_inter = _xy(df, self._cols)
        if self._use_extra:
            preds = self._booster.predict(f_inter, _xy(df, self._extra_cols))
        else:
            preds = self._booster.predict(f_inter)
        # scale per Task 1 finding (rate vs per-bin count) — adjust here if the
        # semantics test showed count units
        return np.clip(preds * self._delta, 0.0, None)


class _SKPoisson(Model):
    def _cols(self, bench: PoissonBench) -> List[str]:
        return bench.partition_cols + ["t"]

    def fit(self, bench: PoissonBench, params: Optional[dict] = None) -> None:
        self._fit_cols = self._cols(bench)
        tr = bench.train
        self._fit_impl(_xy(tr, self._fit_cols), tr["y"].to_numpy(), params or {})

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(self._predict_impl(_xy(df, self._fit_cols)),
                                  dtype=np.float64), 0.0, None)


class LightGBMPoisson(_SKPoisson):
    name = "lgbm_pois"

    def available(self) -> bool:
        try:
            import lightgbm  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, p):
        import lightgbm as lgb
        self._m = lgb.LGBMRegressor(objective="poisson", verbose=-1,
                                    n_estimators=p.get("n_estimators", 100),
                                    max_depth=p.get("max_depth", 5),
                                    learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)

    def _predict_impl(self, X):
        return self._m.predict(X)


class XGBoostPoisson(_SKPoisson):
    name = "xgb_pois"

    def available(self) -> bool:
        try:
            import xgboost  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, p):
        import xgboost as xgb
        self._m = xgb.XGBRegressor(objective="count:poisson",
                                   n_estimators=p.get("n_estimators", 100),
                                   max_depth=p.get("max_depth", 5),
                                   learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)

    def _predict_impl(self, X):
        return self._m.predict(X)


class CatBoostPoisson(_SKPoisson):
    name = "catboost_pois"

    def available(self) -> bool:
        try:
            import catboost  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"iterations": [50, 100, 200], "depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, p):
        import catboost as cb
        self._m = cb.CatBoostRegressor(loss_function="Poisson", verbose=0,
                                       iterations=p.get("iterations", 100),
                                       depth=p.get("depth", 5),
                                       learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)

    def _predict_impl(self, X):
        return self._m.predict(X)


class PoissonOracleModel(Model):
    name = "oracle_pois"

    def fit(self, bench: PoissonBench, params: Optional[dict] = None) -> None:
        self._delta = bench.delta

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["lam_true"].to_numpy() * self._delta


def make_poisson_models(include_oracle: bool = True) -> Dict[str, Model]:
    models: List[Model] = [
        GBDTEPoissonModel("oracle"), GBDTEPoissonModel("wrong"),
        GBDTEPoissonModel("const"),
        LightGBMPoisson(), XGBoostPoisson(), CatBoostPoisson(),
    ]
    if include_oracle:
        models.append(PoissonOracleModel())
    return {m.name: m for m in models}


__all__ = ["GBDTEPoissonModel", "LightGBMPoisson", "XGBoostPoisson",
           "CatBoostPoisson", "PoissonOracleModel", "make_poisson_models"]
```

- [ ] **Step 5: Run all tests, verify PASS; commit**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests -q
git add python/extra_boost_py/experiments/poisson_baselines.py python/extra_boost_py/experiments/stats.py python/tests/test_poisson_baselines.py
git commit -m "Add Poisson wrappers and poisson deviance metric"
```

---

### Task 4: Suite, CLI, full run, reports

**Files:**
- Modify: `python/extra_boost_py/experiments/suites.py` (add `suite_baselines_poisson`, register in `SUITES`)
- Test: `python/tests/test_suites.py` (add quick poisson test)
- Create (generated): `reports/article_experiments/baselines_poisson/`

**Interfaces:**
- Consumes: everything above; `_write_report`, `run_grid`, `_seeds` from `suites.py`/`stats.py`.

- [ ] **Step 1: Add failing test to `python/tests/test_suites.py`**

```python
def test_quick_baselines_poisson(tmp_path: Path):
    out = run_suite("baselines_poisson", tmp_path, seeds=2, quick=True)
    assert (out / "results.csv").exists()
    assert (out / "summary_poisson_dev.md").exists()
```

- [ ] **Step 2: Implement in `suites.py`** (import `POISSON_PRESETS`, `generate_poisson`,
  `make_poisson_models` at top):

```python
def suite_baselines_poisson(out: Path, seeds: int, quick: bool) -> None:
    presets = {k: POISSON_PRESETS[k]
               for k in (["poisson_k8"] if quick else ["poisson_k8", "poisson_k64"])}
    models = make_poisson_models()

    def factory(name: str, seed: int):
        cfg = replace(presets[name], seed=seed,
                      n_objects=60 if quick else presets[name].n_objects,
                      n_bins=8 if quick else presets[name].n_bins)
        return generate_poisson(cfg)

    res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                   bench_names=list(presets), tune_trials=2 if quick else 8)
    _write_report(out, res, {"suite": "baselines_poisson", "git": _git_sha(),
                             "presets": {k: asdict(v) for k, v in presets.items()},
                             "seeds": seeds},
                  {"poisson_dev": False, "rate_rmse": False})
```

Register: `SUITES["baselines_poisson"] = suite_baselines_poisson`.

- [ ] **Step 3: Full test suite green; commit**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests -q
git add python/extra_boost_py/experiments/suites.py python/tests/test_suites.py
git commit -m "Add baselines_poisson suite"
```

- [ ] **Step 4: Full run (background), verify story, commit reports**

```bash
PYTHONPATH=python .venv/bin/python scripts/run_article_experiments.py --suite baselines_poisson --seeds 10
```

Verify: summary tables coherent (oracle best; gbdte_pois ahead of externals on the future
window if drift matters; gbdte_pois_const and _wrong degraded); CD diagram exists. Then:

```bash
git add reports/article_experiments/baselines_poisson
git commit -m "Add Poisson baseline experiment report"
cp -r reports/article_experiments/baselines_poisson ~/prj/gbdte_article_2026/gbdte-main/reports/article_experiments/
```

- [ ] **Step 5: Update `~/prj/gbdte_article_2026/paper_progress.md`** — add Poisson row to
  the "New experimental evidence" section with headline numbers.

---

## Self-Review Notes

- Spec coverage: gate→Task1, generator→Task2, wrappers+metrics→Task3, suite+run→Task4. Spec's
  "Poisson regime map" explicitly out of scope — no task, correct.
- Type consistency: `PoissonBench.event_train()` tuple order matches Task 3 usage;
  `make_poisson_models` keys match suite expectations; `evaluate` duck-types on
  `bench.task`/`bench.delta`.
- Risk: Task 1 may reveal count-unit predictions; Task 3's `predict` has the single marked
  line to adjust (`preds * self._delta` → `preds`).

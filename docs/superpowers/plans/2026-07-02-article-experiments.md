# Article Experiments (E1–E7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproducible experiment layer (parametric benchmark family, baseline matrix, multi-seed statistics with CD diagrams, role-separability statistic, auto role assignment) producing paper-ready tables/figures under `reports/article_experiments/`.

**Architecture:** New consumer package `python/extra_boost_py/experiments/` (benchgen → baselines → rolestat → stats → suites) plus CLI `scripts/run_article_experiments.py`. The Go engine and existing pipeline are not modified; GBDTE is used via `extra_boost_py.booster.ExtraBooster`.

**Tech Stack:** Python 3 (uv venv), numpy/pandas/scipy, scikit-learn, lightgbm (`linear_tree`), xgboost, catboost, scikit-posthocs, matplotlib (vector PDF), pytest. Go 1.18 via existing CGO bridge.

## Global Constraints

- All randomness goes through `np.random.default_rng(seed)`; every report embeds config JSON + git SHA.
- Figures are vector PDF (reviewer demand); tables are emitted as md + LaTeX + csv.
- Baseline wrappers degrade to "skipped" (recorded in report) if their package is missing; they never abort a suite.
- Tuning: identical random-search budget per model, validation = last-by-`t` 25% of the train window; the test window is never touched during tuning.
- Tests live in `python/tests/`; run via `PYTHONPATH=python .venv/bin/pytest python/tests -q`.
- Commit after each task with a present-tense subject (repo convention).

---

### Task 1: Environment + benchgen (E1)

**Files:**
- Create: `.venv` (uv), `python/extra_boost_py/experiments/__init__.py` (empty), `python/extra_boost_py/experiments/benchgen.py`
- Test: `python/tests/test_benchgen.py`

**Interfaces:**
- Produces: `BenchConfig` (frozen dataclass: task:str="mse", k_groups:int=8, basis:str="fourier", omega:float=50.0, rho:float=1.0, drift:float=1.0, noise:float=0.01, cut:float=0.5, n_rows:int=4000, seed:int=0), `Bench` (dataclass: df:pd.DataFrame, partition_cols:list[str], extra_cols:list[str], task:str, cut:float; properties `train`/`test` returning row-subset DataFrames by `t` vs `cut`), `generate(cfg: BenchConfig) -> Bench`, `PRESETS: dict[str, BenchConfig]` with keys `mse_k8`, `mse_k128`, `regime_base` (task=mse, k=32, basis=fourier, omega=8.0).

- [ ] **Step 1: Create venv and install deps**

```bash
cd ~/prj/extra_bridged_boosting
uv venv .venv
uv pip install --python .venv/bin/python numpy pandas scipy scikit-learn matplotlib lightgbm xgboost catboost scikit-posthocs pytest
```

- [ ] **Step 2: Build the Go shared library and smoke-test the bridge**

```bash
PYTHONPATH=python .venv/bin/python -c "from extra_boost_py.go_lib import build_shared; build_shared()"
PYTHONPATH=python .venv/bin/python scripts/run_smoke_tests.py
```
Expected: smoke tests pass. If go1.18 fails to build, stop and report (spec: fail fast).

- [ ] **Step 3: Write failing tests**

`python/tests/test_benchgen.py`:

```python
import numpy as np
from extra_boost_py.experiments.benchgen import BenchConfig, generate, PRESETS


def test_shapes_and_columns():
    b = generate(BenchConfig(k_groups=8, n_rows=1000, seed=1))
    assert len(b.df) == 1000
    assert b.partition_cols == ["f_0", "f_1", "f_2", "f_m"]
    assert b.extra_cols == ["e_0", "e_1", "e_2", "e_3"]  # fourier basis
    assert {"t", "y", "group", "oracle"} <= set(b.df.columns)
    assert set(np.unique(b.df["group"])) <= set(range(8))


def test_determinism():
    a = generate(BenchConfig(seed=7, n_rows=500))
    b = generate(BenchConfig(seed=7, n_rows=500))
    assert a.df.equals(b.df)


def test_temporal_split():
    b = generate(BenchConfig(n_rows=1000, cut=0.5, seed=0))
    assert (b.train["t"] < 0.5).all() and (b.test["t"] >= 0.5).all()
    assert len(b.train) + len(b.test) == 1000


def test_rho1_pure_roles():
    # at rho=1 target is exactly phi(t)^T beta_g + noise: per-group regression on the
    # e-columns must fit near-perfectly
    b = generate(BenchConfig(rho=1.0, noise=0.0, n_rows=2000, seed=3))
    df = b.df
    for g in range(4):
        part = df[df["group"] == g]
        E = part[b.extra_cols].to_numpy()
        coef, *_ = np.linalg.lstsq(E, part["y"].to_numpy(), rcond=None)
        resid = part["y"].to_numpy() - E @ coef
        assert np.abs(resid).max() < 1e-8


def test_logloss_has_bernoulli_target_and_oracle():
    b = generate(BenchConfig(task="logloss", noise=0.0, n_rows=1000, seed=2))
    assert set(np.unique(b.df["y"])) <= {0.0, 1.0}
    assert np.isfinite(b.df["oracle"]).all()
```

- [ ] **Step 4: Run tests, verify failure**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests/test_benchgen.py -q
```
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 5: Implement `benchgen.py`**

```python
"""E1: parametric benchmark family generalizing the paper's MSE/LogLoss generators."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchConfig:
    task: str = "mse"            # "mse" | "logloss"
    k_groups: int = 8
    basis: str = "fourier"       # "linear" | "fourier"
    omega: float = 50.0
    rho: float = 1.0             # role purity in [0, 1]
    drift: float = 1.0           # scale of time-dependent coefficient part
    noise: float = 0.01
    cut: float = 0.5
    n_rows: int = 4000
    seed: int = 0


@dataclass
class Bench:
    df: pd.DataFrame
    partition_cols: List[str]
    extra_cols: List[str]
    task: str
    cut: float
    config: BenchConfig | None = None

    @property
    def train(self) -> pd.DataFrame:
        return self.df[self.df["t"] < self.cut]

    @property
    def test(self) -> pd.DataFrame:
        return self.df[self.df["t"] >= self.cut]


def _phi(t: np.ndarray, basis: str, omega: float) -> np.ndarray:
    if basis == "linear":
        return np.column_stack([np.ones_like(t), t])
    if basis == "fourier":
        return np.column_stack([np.ones_like(t), t, np.sin(omega * t), np.cos(omega * t)])
    raise ValueError(f"unknown basis {basis!r}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(cfg: BenchConfig) -> Bench:
    rng = np.random.default_rng(cfg.seed)
    n, k = cfg.n_rows, cfg.k_groups
    n_bits = max(1, int(np.ceil(np.log2(k))))

    t = rng.random(n)
    group = rng.integers(0, k, size=n)
    bits = ((group[:, None] >> np.arange(n_bits)) & 1).astype(np.float64)
    m = rng.random(n)  # mixed nuisance feature: violates the role split for rho < 1

    E = _phi(t, cfg.basis, cfg.omega)
    d = E.shape[1]

    beta = rng.standard_normal((k, d))
    beta[:, 1:] *= cfg.drift                      # non-constant part scaled by drift
    score_pure = np.einsum("ij,ij->i", E, beta[group])

    # impure component: coefficients vary smoothly with m instead of with the group
    beta_m = np.sin(2.0 * np.pi * (np.arange(d)[None, :] + 1.0) * m[:, None])
    beta_m[:, 1:] *= cfg.drift
    score_mixed = np.einsum("ij,ij->i", E, beta_m)

    score = cfg.rho * score_pure + (1.0 - cfg.rho) * score_mixed

    if cfg.task == "mse":
        y = score + cfg.noise * rng.standard_normal(n)
        oracle = score
    elif cfg.task == "logloss":
        p = _sigmoid(score)
        y = (rng.random(n) < p).astype(np.float64)
        oracle = score
    else:
        raise ValueError(f"unknown task {cfg.task!r}")

    data = {f"f_{i}": bits[:, i] for i in range(n_bits)}
    data["f_m"] = m
    data["t"] = t
    for j in range(d):
        data[f"e_{j}"] = E[:, j]
    data["y"] = y
    data["group"] = group.astype(np.int64)
    data["oracle"] = oracle

    df = pd.DataFrame(data)
    return Bench(
        df=df,
        partition_cols=[f"f_{i}" for i in range(n_bits)] + ["f_m"],
        extra_cols=[f"e_{j}" for j in range(d)],
        task=cfg.task,
        cut=cfg.cut,
        config=cfg,
    )


PRESETS = {
    "mse_k8": BenchConfig(task="mse", k_groups=8, basis="fourier", omega=50.0,
                          rho=1.0, drift=1.0, noise=0.01, n_rows=4000),
    "mse_k128": BenchConfig(task="mse", k_groups=128, basis="fourier", omega=50.0,
                            rho=1.0, drift=1.0, noise=0.01, n_rows=10000),
    "regime_base": BenchConfig(task="mse", k_groups=32, basis="fourier", omega=8.0,
                               rho=1.0, drift=1.0, noise=0.05, n_rows=4000),
}

__all__ = ["BenchConfig", "Bench", "generate", "PRESETS"]
```

Note: `test_rho1_pure_roles` checks groups 0..3 only so it also passes with k=8 defaults.

- [ ] **Step 6: Run tests, verify pass**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests/test_benchgen.py -q
```
Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add python/extra_boost_py/experiments python/tests/test_benchgen.py
git commit -m "Add parametric benchmark family generator (E1)"
```

---

### Task 2: Classical LogLoss adapter with Bayes oracle

**Files:**
- Create: `python/extra_boost_py/experiments/classical_bench.py`
- Test: `python/tests/test_classical_bench.py`

**Interfaces:**
- Consumes: `Bench` from Task 1; `generate_classical_dataset`, `ClassicalFeatureSpec` from `extra_boost_py.classical_dataset`.
- Produces: `classical_bench(n_rows: int = 10000, seed: int = 0, cut: float = 0.4, alpha: float = 0.3) -> Bench` — task="logloss", partition_cols=`["f_0".."f_16"]`, extra_cols=`["e_0","e_1"]` (=[1,t]), df has exact Bayes `oracle` column.

The generator makes features conditionally independent given the label, so the Bayes
log-odds are analytic. For feature spec s at time t: `P(f=1|y=1)=beta_s`,
`P(f=1|y=0)=gamma_s(t)=((1/(alpha*lift)-1)/(1/alpha-1))*beta_s` with `lift=slope*t+intercept`.
Oracle log-odds = `log(alpha/(1-alpha)) + sum_s [ f*log(beta_s/gamma_s) + (1-f)*log((1-beta_s)/(1-gamma_s)) ]`.

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from sklearn.metrics import log_loss
from extra_boost_py.experiments.classical_bench import classical_bench


def test_structure():
    b = classical_bench(n_rows=2000, seed=0)
    assert b.task == "logloss"
    assert len(b.partition_cols) == 17 and b.extra_cols == ["e_0", "e_1"]
    assert (b.train["t"] < 0.4).all() and (b.test["t"] >= 0.4).all()


def test_oracle_beats_marginal():
    b = classical_bench(n_rows=20000, seed=1)
    y = b.df["y"].to_numpy()
    p_oracle = 1.0 / (1.0 + np.exp(-b.df["oracle"].to_numpy()))
    ll_oracle = log_loss(y, p_oracle)
    ll_marginal = log_loss(y, np.full_like(y, y.mean()))
    assert ll_oracle < ll_marginal - 0.02
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_classical_bench.py -q`

- [ ] **Step 3: Implement**

```python
"""Adapter: the paper's classical LogLoss benchmark as a Bench, with exact Bayes oracle."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..classical_dataset import ClassicalFeatureSpec, generate_classical_dataset
from .benchgen import Bench

_ASCEND = ClassicalFeatureSpec(slope=0.75, intercept=0.5, beta=0.2)
_DESCEND = ClassicalFeatureSpec(slope=-0.75, intercept=1.25, beta=0.1)


def _specs() -> list[ClassicalFeatureSpec]:
    specs = [_ASCEND, _DESCEND]
    for idx in range(15):
        specs.append(_DESCEND if idx < 7 else _ASCEND)
    return specs


def _oracle_logodds(features: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    denom = 1.0 / alpha - 1.0
    score = np.full(t.shape, np.log(alpha / (1.0 - alpha)))
    for col, s in enumerate(_specs()):
        lift = s.slope * t + s.intercept
        gamma = ((1.0 / (alpha * lift) - 1.0) / denom) * s.beta
        f = features[:, col]
        score += f * np.log(s.beta / gamma) + (1.0 - f) * np.log((1.0 - s.beta) / (1.0 - gamma))
    return score


def classical_bench(n_rows: int = 10000, seed: int = 0, cut: float = 0.4,
                    alpha: float = 0.3) -> Bench:
    ds = generate_classical_dataset(n_rows, alpha=alpha, seed=seed)
    n_feat = ds.features_inter.shape[1]
    data = {f"f_{i}": ds.features_inter[:, i] for i in range(n_feat)}
    data["t"] = ds.time
    data["e_0"] = np.ones_like(ds.time)
    data["e_1"] = ds.time
    data["y"] = ds.target
    data["oracle"] = _oracle_logodds(ds.features_inter, ds.time, alpha)
    return Bench(
        df=pd.DataFrame(data),
        partition_cols=[f"f_{i}" for i in range(n_feat)],
        extra_cols=["e_0", "e_1"],
        task="logloss",
        cut=cut,
    )


__all__ = ["classical_bench"]
```

- [ ] **Step 4: Run, verify PASS**, then commit:

```bash
git add python/extra_boost_py/experiments/classical_bench.py python/tests/test_classical_bench.py
git commit -m "Add classical LogLoss bench adapter with exact Bayes oracle"
```

---

### Task 3: Role-separability statistic + auto roles (E4, E5)

**Files:**
- Create: `python/extra_boost_py/experiments/rolestat.py`
- Test: `python/tests/test_rolestat.py`

**Interfaces:**
- Consumes: `Bench`, `BenchConfig`, `generate`.
- Produces: `drift_score(x: np.ndarray, t: np.ndarray) -> float` (in [0.5, 1]); `stability_score(x, y, t) -> float` (≥0); `separability_index(bench: Bench) -> float`; `auto_roles(bench: Bench, drift_threshold: float = 0.6) -> tuple[list[str], list[str]]` returning (partition_cols, leaf_cols) chosen from `bench.partition_cols + ["t"]`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from extra_boost_py.experiments.benchgen import BenchConfig, generate
from extra_boost_py.experiments.rolestat import (
    drift_score, separability_index, auto_roles)


def test_drift_score_detects_time_feature():
    rng = np.random.default_rng(0)
    t = rng.random(4000)
    assert drift_score(t + 0.1 * rng.standard_normal(4000), t) > 0.9
    assert drift_score(rng.random(4000), t) < 0.55


def test_separability_monotone_in_rho():
    vals = []
    for rho in (0.0, 0.5, 1.0):
        idx = np.mean([
            separability_index(generate(BenchConfig(
                k_groups=8, rho=rho, drift=1.0, omega=8.0,
                noise=0.05, n_rows=4000, seed=s)))
            for s in range(3)
        ])
        vals.append(idx)
    assert vals[0] < vals[1] < vals[2]


def test_auto_roles_recovers_design():
    b = generate(BenchConfig(k_groups=8, rho=1.0, omega=8.0, n_rows=4000, seed=0))
    part, leaf = auto_roles(b)
    assert "t" in leaf
    assert {"f_0", "f_1", "f_2"} <= set(part)
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_rolestat.py -q`

- [ ] **Step 3: Implement**

```python
"""E4/E5: data-computable role-separability statistics and automatic role assignment."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .benchgen import Bench


def drift_score(x: np.ndarray, t: np.ndarray) -> float:
    """How much a single feature carries time signal: folded AUC of x vs late-half."""
    late = (t > np.median(t)).astype(int)
    if np.unique(x).size < 2:
        return 0.5
    auc = roc_auc_score(late, x)
    return float(max(auc, 1.0 - auc))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def stability_score(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """Cross-time stability of the feature-target relation: min |corr| over time halves
    if the sign agrees, else 0."""
    med = np.median(t)
    early, late = t <= med, t > med
    c1, c2 = _corr(x[early], y[early]), _corr(x[late], y[late])
    if c1 * c2 <= 0.0:
        return 0.0
    return float(min(abs(c1), abs(c2)))


def separability_index(bench: Bench) -> float:
    """Mean stability of binary partition candidates; tracks role purity (rho)."""
    df = bench.df
    y, t = df["y"].to_numpy(), df["t"].to_numpy()
    scores = []
    for col in bench.partition_cols:
        x = df[col].to_numpy()
        if np.unique(x).size <= 2:  # binary candidates only
            scores.append(stability_score(x, y, t))
    return float(np.mean(scores)) if scores else 0.0


def auto_roles(bench: Bench, drift_threshold: float = 0.6) -> Tuple[List[str], List[str]]:
    """Assign candidate columns to partition vs leaf blocks by drift score."""
    df = bench.df
    t = df["t"].to_numpy()
    partition, leaf = [], ["t"]
    for col in bench.partition_cols:
        if drift_score(df[col].to_numpy(), t) >= drift_threshold:
            leaf.append(col)
        else:
            partition.append(col)
    return partition, leaf


__all__ = ["drift_score", "stability_score", "separability_index", "auto_roles"]
```

- [ ] **Step 4: Run, verify PASS**, commit:

```bash
git add python/extra_boost_py/experiments/rolestat.py python/tests/test_rolestat.py
git commit -m "Add role-separability statistic and automatic role assignment (E4, E5)"
```

---

### Task 4: Baseline model wrappers (E7)

**Files:**
- Create: `python/extra_boost_py/experiments/baselines.py`
- Test: `python/tests/test_baselines.py`

**Interfaces:**
- Consumes: `Bench`; `ExtraBooster`, `BoosterParams` from `extra_boost_py.booster`; `auto_roles` from Task 3.
- Produces: `make_models(task: str, include_oracle: bool = True) -> dict[str, Model]`; `class Model` with `name: str`, `fit(bench: Bench, params: dict | None = None) -> None`, `predict(df: pd.DataFrame) -> np.ndarray` (raw score for logloss, value for mse), `available() -> bool`, `param_grid() -> dict[str, list]`. GBDTE variants: `gbdte` (designed roles), `gbdte_auto`, `gbdte_wrong` (blocks swapped), `gbdte_all_in_leaf` (partition features added to the leaf basis), `gbdte_const` (extra=[1] only → constant leaves, "no separation"). External: `lgbm_linear`, `lgbm`, `xgb`, `catboost`; references: `oracle` (logloss), `group_linear` (mse).

- [ ] **Step 1: Write failing smoke test**

```python
import numpy as np
import pytest
from extra_boost_py.experiments.benchgen import BenchConfig, generate
from extra_boost_py.experiments.baselines import make_models


@pytest.mark.parametrize("task", ["mse", "logloss"])
def test_all_models_fit_predict(task):
    b = generate(BenchConfig(task=task, k_groups=4, omega=8.0, n_rows=600, seed=0))
    models = make_models(task)
    assert "gbdte" in models and "lgbm_linear" in models
    for name, model in models.items():
        if not model.available():
            continue
        model.fit(b)
        pred = model.predict(b.test)
        assert pred.shape == (len(b.test),)
        assert np.isfinite(pred).all(), name
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_baselines.py -q`

- [ ] **Step 3: Implement `baselines.py`**

```python
"""E7: unified model wrappers. predict() returns raw score (logloss) or value (mse)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..booster import BoosterParams, ExtraBooster
from .benchgen import Bench
from .rolestat import auto_roles


class Model:
    name = "base"

    def available(self) -> bool:
        return True

    def param_grid(self) -> dict:
        return {}

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


def _xy(df: pd.DataFrame, cols: List[str]):
    return np.ascontiguousarray(df[cols].to_numpy(dtype=np.float64))


class GBDTEModel(Model):
    def __init__(self, roles: str = "oracle"):
        self.roles = roles
        self.name = {"oracle": "gbdte", "auto": "gbdte_auto", "wrong": "gbdte_wrong",
                     "all_in_leaf": "gbdte_all_in_leaf", "const": "gbdte_const"}[roles]
        self._booster = None
        self._inter_cols: List[str] = []
        self._extra_cols: List[str] = []

    def param_grid(self) -> dict:
        return {"n_stages": [24, 48, 96], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _blocks(self, bench: Bench):
        p, e = bench.partition_cols, bench.extra_cols
        if self.roles == "oracle":
            return p, e
        if self.roles == "wrong":
            return e, ["e_0"] + p          # blocks swapped (e_0 is the constant term)
        if self.roles == "all_in_leaf":
            return p, e + p                # tg_072 failure mode
        if self.roles == "const":
            return p, ["e_0"]              # constant leaves, no separation
        if self.roles == "auto":
            part, leaf = auto_roles(bench)
            # leaf candidates are raw columns; prepend the constant term
            return part if part else ["e_0"], ["e_0"] + leaf
        raise ValueError(self.roles)

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        self._inter_cols, self._extra_cols = self._blocks(bench)
        p = params or {}
        bp = BoosterParams(loss=bench.task,
                           n_stages=p.get("n_stages", 48),
                           max_depth=p.get("max_depth", 5),
                           learning_rate=p.get("learning_rate", 0.1))
        tr = bench.train
        self._booster = ExtraBooster.train(
            _xy(tr, self._inter_cols), _xy(tr, self._extra_cols),
            np.ascontiguousarray(tr["y"].to_numpy(dtype=np.float64)), bp)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._booster.predict(_xy(df, self._inter_cols), _xy(df, self._extra_cols))


class _SKStyleModel(Model):
    """Shared logic for external boosters: X = partition candidates + t."""

    def _cols(self, bench: Bench) -> List[str]:
        return bench.partition_cols + ["t"]

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        self._fit_cols = self._cols(bench)
        tr = bench.train
        self._fit_impl(_xy(tr, self._fit_cols), tr["y"].to_numpy(), bench.task,
                       params or {})

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict_impl(_xy(df, self._fit_cols))


class LightGBMModel(_SKStyleModel):
    def __init__(self, linear_tree: bool):
        self.linear_tree = linear_tree
        self.name = "lgbm_linear" if linear_tree else "lgbm"

    def available(self) -> bool:
        try:
            import lightgbm  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, task, p):
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if task == "mse" else lgb.LGBMClassifier
        self._m = cls(linear_tree=self.linear_tree, verbose=-1,
                      n_estimators=p.get("n_estimators", 100),
                      max_depth=p.get("max_depth", 5),
                      learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)
        self._task = task

    def _predict_impl(self, X):
        if self._task == "mse":
            return self._m.predict(X)
        return self._m.predict_proba(X, raw_score=True)


class XGBoostModel(_SKStyleModel):
    name = "xgb"

    def available(self) -> bool:
        try:
            import xgboost  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, task, p):
        import xgboost as xgb
        cls = xgb.XGBRegressor if task == "mse" else xgb.XGBClassifier
        self._m = cls(n_estimators=p.get("n_estimators", 100),
                      max_depth=p.get("max_depth", 5),
                      learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)
        self._task = task

    def _predict_impl(self, X):
        if self._task == "mse":
            return self._m.predict(X)
        return self._m.predict(X, output_margin=True)


class CatBoostModel(_SKStyleModel):
    name = "catboost"

    def available(self) -> bool:
        try:
            import catboost  # noqa: F401
            return True
        except ImportError:
            return False

    def param_grid(self) -> dict:
        return {"iterations": [50, 100, 200], "depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.3]}

    def _fit_impl(self, X, y, task, p):
        import catboost as cb
        cls = cb.CatBoostRegressor if task == "mse" else cb.CatBoostClassifier
        self._m = cls(verbose=0, iterations=p.get("iterations", 100),
                      depth=p.get("depth", 5),
                      learning_rate=p.get("learning_rate", 0.1))
        self._m.fit(X, y)
        self._task = task

    def _predict_impl(self, X):
        if self._task == "mse":
            return self._m.predict(X)
        return self._m.predict(X, prediction_type="RawFormulaVal")


class OracleModel(Model):
    """Bayes-optimal score provided by the generator (logloss benches only)."""
    name = "oracle"

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        pass

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["oracle"].to_numpy()


class GroupLinearModel(Model):
    """Per-true-group least squares on the extrapolation basis (mse upper bound)."""
    name = "group_linear"

    def fit(self, bench: Bench, params: Optional[dict] = None) -> None:
        self._extra_cols = bench.extra_cols
        self._coef = {}
        tr = bench.train
        for g, part in tr.groupby("group"):
            E = part[self._extra_cols].to_numpy()
            self._coef[g], *_ = np.linalg.lstsq(E, part["y"].to_numpy(), rcond=None)
        self._default = np.mean(list(self._coef.values()), axis=0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        E = df[self._extra_cols].to_numpy()
        coefs = np.array([self._coef.get(g, self._default) for g in df["group"]])
        return np.einsum("ij,ij->i", E, coefs)


def make_models(task: str, include_oracle: bool = True) -> Dict[str, Model]:
    models: List[Model] = [
        GBDTEModel("oracle"), GBDTEModel("auto"), GBDTEModel("wrong"),
        GBDTEModel("all_in_leaf"), GBDTEModel("const"),
        LightGBMModel(linear_tree=True), LightGBMModel(linear_tree=False),
        XGBoostModel(), CatBoostModel(),
    ]
    if task == "mse":
        models.append(GroupLinearModel())
    elif include_oracle:
        models.append(OracleModel())
    return {m.name: m for m in models}


__all__ = ["Model", "GBDTEModel", "LightGBMModel", "XGBoostModel", "CatBoostModel",
           "OracleModel", "GroupLinearModel", "make_models"]
```

- [ ] **Step 4: Run, verify PASS** (external packages installed in Task 1, so nothing is skipped), commit:

```bash
git add python/extra_boost_py/experiments/baselines.py python/tests/test_baselines.py
git commit -m "Add unified baseline wrappers incl. LightGBM linear_tree and GBDTE ablations (E7)"
```

---

### Task 5: Statistics runner, tuning, Friedman/Nemenyi + CD diagram (E6)

**Files:**
- Create: `python/extra_boost_py/experiments/stats.py`
- Test: `python/tests/test_stats.py`

**Interfaces:**
- Consumes: `Bench`, `Model`, `make_models`.
- Produces: `evaluate(model, bench) -> dict[str, float]` (mse → {"rmse"}, logloss → {"auc","logloss"}); `tune(model, bench, n_trials=8, seed=0) -> dict` (random search on `param_grid()`, validated on last-by-t 25% of train); `run_grid(models: dict, benches: dict[str, Bench], seeds: list[int], bench_factory=None) -> pd.DataFrame` tidy columns (bench, model, seed, metric, value); `summary_table(df, metric) -> pd.DataFrame` (mean±std pivot); `friedman_nemenyi(df, metric, higher_is_better) -> dict`; `cd_diagram(df, metric, higher_is_better, out_pdf: Path)`.

- [ ] **Step 1: Write failing tests** (statistics paths on synthetic metric tables; no model training)

```python
import numpy as np
import pandas as pd
from pathlib import Path
from extra_boost_py.experiments.stats import summary_table, friedman_nemenyi, cd_diagram


def _fake_results():
    rng = np.random.default_rng(0)
    rows = []
    for bench in ["b1", "b2", "b3", "b4", "b5", "b6"]:
        for model, mu in [("good", 0.1), ("mid", 0.2), ("bad", 0.4)]:
            for seed in range(5):
                rows.append(dict(bench=bench, model=model, seed=seed,
                                 metric="rmse", value=mu + 0.01 * rng.standard_normal()))
    return pd.DataFrame(rows)


def test_summary_table():
    tab = summary_table(_fake_results(), "rmse")
    assert set(tab.columns) == {"good", "mid", "bad"}
    assert "±" in tab.iloc[0, 0]


def test_friedman_detects_difference():
    res = friedman_nemenyi(_fake_results(), "rmse", higher_is_better=False)
    assert res["p_value"] < 0.01
    assert res["avg_ranks"]["good"] < res["avg_ranks"]["bad"]


def test_cd_diagram_writes_pdf(tmp_path: Path):
    out = tmp_path / "cd.pdf"
    cd_diagram(_fake_results(), "rmse", False, out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_stats.py -q`

- [ ] **Step 3: Implement `stats.py`**

```python
"""E6: multi-seed evaluation, uniform tuning, Friedman/Nemenyi and CD diagram."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import log_loss, roc_auc_score

from .benchgen import Bench
from .baselines import Model


def evaluate(model: Model, bench: Bench) -> Dict[str, float]:
    test = bench.test
    pred = model.predict(test)
    y = test["y"].to_numpy()
    if bench.task == "mse":
        return {"rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}
    prob = 1.0 / (1.0 + np.exp(-pred))
    return {"auc": float(roc_auc_score(y, prob)),
            "logloss": float(log_loss(y, np.clip(prob, 1e-12, 1 - 1e-12)))}


def _val_split(bench: Bench) -> tuple[Bench, Bench]:
    """Last-by-t 25% of the train window becomes validation."""
    tr = bench.train
    q = tr["t"].quantile(0.75)
    inner = Bench(df=tr[tr["t"] < q], partition_cols=bench.partition_cols,
                  extra_cols=bench.extra_cols, task=bench.task, cut=float(q))
    val = Bench(df=tr, partition_cols=bench.partition_cols,
                extra_cols=bench.extra_cols, task=bench.task, cut=float(q))
    return inner, val


def tune(model: Model, bench: Bench, n_trials: int = 8, seed: int = 0) -> dict:
    grid = model.param_grid()
    if not grid:
        return {}
    rng = np.random.default_rng(seed)
    combos = list(itertools.product(*grid.values()))
    rng.shuffle(combos)
    key = "rmse" if bench.task == "mse" else "logloss"
    inner, val = _val_split(bench)
    best, best_score = {}, np.inf
    for combo in combos[:n_trials]:
        params = dict(zip(grid.keys(), combo))
        model.fit(inner, params)
        score = evaluate(model, val)[key]
        if score < best_score:
            best, best_score = params, score
    return best


def run_grid(models: Dict[str, Model], benches: Dict[str, Bench] | None,
             seeds: List[int],
             bench_factory: Optional[Callable[[str, int], Bench]] = None,
             bench_names: Optional[List[str]] = None,
             tune_trials: int = 8) -> pd.DataFrame:
    """benches: fixed Bench per name, re-generated per seed via bench_factory if given."""
    names = bench_names or list(benches.keys())
    rows = []
    tuned: Dict[tuple, dict] = {}
    for bname in names:
        for seed in seeds:
            bench = bench_factory(bname, seed) if bench_factory else benches[bname]
            for mname, model in models.items():
                if not model.available():
                    rows.append(dict(bench=bname, model=mname, seed=seed,
                                     metric="skipped", value=np.nan))
                    continue
                if (bname, mname) not in tuned:
                    tuned[(bname, mname)] = tune(model, bench, tune_trials, seed)
                model.fit(bench, tuned[(bname, mname)])
                for metric, value in evaluate(model, bench).items():
                    rows.append(dict(bench=bname, model=mname, seed=seed,
                                     metric=metric, value=value))
    return pd.DataFrame(rows)


def summary_table(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = results[results["metric"] == metric]
    agg = sub.groupby(["bench", "model"])["value"].agg(["mean", "std"]).reset_index()
    agg["cell"] = agg.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    return agg.pivot(index="bench", columns="model", values="cell")


def _rank_table(results: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    sub = results[results["metric"] == metric]
    perf = sub.groupby(["bench", "seed", "model"])["value"].mean().unstack("model")
    ranks = perf.rank(axis=1, ascending=not higher_is_better)
    return ranks


def friedman_nemenyi(results: pd.DataFrame, metric: str, higher_is_better: bool) -> dict:
    ranks = _rank_table(results, metric, higher_is_better)
    cols = list(ranks.columns)
    stat, p = sps.friedmanchisquare(*[ranks[c].to_numpy() for c in cols])
    out = {"statistic": float(stat), "p_value": float(p),
           "avg_ranks": ranks.mean().to_dict(), "n_blocks": len(ranks)}
    try:
        import scikit_posthocs as sp
        sub = results[results["metric"] == metric]
        perf = sub.groupby(["bench", "seed", "model"])["value"].mean().reset_index()
        out["nemenyi_p"] = sp.posthoc_nemenyi_friedman(
            perf.pivot(index=["bench", "seed"], columns="model", values="value")
        ).to_dict()
    except ImportError:
        out["nemenyi_p"] = None
    return out


def cd_diagram(results: pd.DataFrame, metric: str, higher_is_better: bool,
               out_pdf: Path) -> None:
    ranks = _rank_table(results, metric, higher_is_better)
    avg = ranks.mean().sort_values()
    k, n = len(avg), len(ranks)
    q_alpha = sps.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2.0)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    fig, ax = plt.subplots(figsize=(7, 0.6 * k + 1.2))
    y = np.arange(k)[::-1]
    ax.hlines(y, xmin=1, xmax=avg.to_numpy(), color="0.7", lw=1)
    ax.plot(avg.to_numpy(), y, "o", color="C0")
    for yi, (name, r) in zip(y, avg.items()):
        ax.annotate(f"  {name} ({r:.2f})", (r, yi), va="center", fontsize=9)
    ax.plot([1, 1 + cd], [k - 0.4] * 2, lw=3, color="C3")
    ax.annotate(f"CD = {cd:.2f}", (1, k - 0.15), color="C3", fontsize=9)
    ax.set_xlabel(f"average rank ({metric})")
    ax.set_yticks([])
    ax.set_xlim(0.8, k + 0.8)
    ax.set_ylim(-0.8, k)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


__all__ = ["evaluate", "tune", "run_grid", "summary_table",
           "friedman_nemenyi", "cd_diagram"]
```

- [ ] **Step 4: Run, verify PASS**, commit:

```bash
git add python/extra_boost_py/experiments/stats.py python/tests/test_stats.py
git commit -m "Add multi-seed stats runner with Friedman/Nemenyi and CD diagram (E6)"
```

---

### Task 6: Suites + CLI (E2 regime map, report writing)

**Files:**
- Create: `python/extra_boost_py/experiments/suites.py`, `scripts/run_article_experiments.py`
- Test: `python/tests/test_suites.py` (tiny smoke: one suite with n_rows=400, 2 seeds, 2 models)

**Interfaces:**
- Consumes: everything above.
- Produces: `run_suite(name: str, out_root: Path, seeds: int = 10, quick: bool = False) -> Path`; suites `baselines_mse`, `baselines_logloss`, `regime_map`, `rolestat_validation`, `auto_roles`. Each writes `results.csv`, `summary_<metric>.md`, `table_<metric>.tex`, `stats_<metric>.json`, figures `*.pdf`, and `run_meta.json` (config + git SHA).

Key contents (implement exactly):

```python
"""Named experiment suites producing paper-ready reports."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchgen import Bench, BenchConfig, PRESETS, generate
from .classical_bench import classical_bench
from .baselines import make_models
from .rolestat import separability_index
from .stats import cd_diagram, friedman_nemenyi, run_grid, summary_table

RHO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
DRIFT_GRID = [0.0, 0.5, 1.0, 2.0]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_report(out: Path, results: pd.DataFrame, meta: dict,
                  higher_is_better: Dict[str, bool]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    for metric, hib in higher_is_better.items():
        if not (results["metric"] == metric).any():
            continue
        tab = summary_table(results, metric)
        tab.to_markdown(out / f"summary_{metric}.md")
        tab.to_latex(out / f"table_{metric}.tex")
        n_models = results[results["metric"] == metric]["model"].nunique()
        if n_models >= 3:
            st = friedman_nemenyi(results, metric, hib)
            (out / f"stats_{metric}.json").write_text(json.dumps(st, indent=2))
            cd_diagram(results, metric, hib, out / f"cd_{metric}.pdf")


def _seeds(n: int) -> list[int]:
    return list(range(n))


def suite_baselines_mse(out: Path, seeds: int, quick: bool) -> None:
    presets = {k: PRESETS[k] for k in (["mse_k8"] if quick else ["mse_k8", "mse_k128"])}
    models = make_models("mse")

    def factory(name: str, seed: int) -> Bench:
        cfg = replace(presets[name], seed=seed, n_rows=400 if quick else presets[name].n_rows)
        return generate(cfg)

    res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                   bench_names=list(presets), tune_trials=2 if quick else 8)
    _write_report(out, res, {"suite": "baselines_mse", "git": _git_sha(),
                             "presets": {k: asdict(v) for k, v in presets.items()},
                             "seeds": seeds}, {"rmse": False})


def suite_baselines_logloss(out: Path, seeds: int, quick: bool) -> None:
    models = make_models("logloss")

    def factory(name: str, seed: int) -> Bench:
        return classical_bench(n_rows=400 if quick else 10000, seed=seed)

    res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                   bench_names=["classical"], tune_trials=2 if quick else 8)
    _write_report(out, res, {"suite": "baselines_logloss", "git": _git_sha(),
                             "seeds": seeds}, {"auc": True, "logloss": False})


def _regime_results(seeds: int, quick: bool, model_names: list[str]) -> pd.DataFrame:
    base = PRESETS["regime_base"]
    all_models = make_models("mse")
    models = {k: all_models[k] for k in model_names}
    names = [f"rho{rho}_drift{dr}" for rho in RHO_GRID for dr in DRIFT_GRID]
    if quick:
        names = names[:4]

    def factory(name: str, seed: int) -> Bench:
        rho = float(name.split("_")[0][3:])
        dr = float(name.split("_")[1][5:])
        cfg = replace(base, rho=rho, drift=dr, seed=seed,
                      n_rows=400 if quick else base.n_rows)
        return generate(cfg)

    return run_grid(models, None, _seeds(seeds), bench_factory=factory,
                    bench_names=names, tune_trials=0)


def suite_regime_map(out: Path, seeds: int, quick: bool) -> None:
    res = _regime_results(seeds, quick, ["gbdte", "lgbm"])
    _write_report(out, res, {"suite": "regime_map", "git": _git_sha(), "seeds": seeds,
                             "rho_grid": RHO_GRID, "drift_grid": DRIFT_GRID},
                  {"rmse": False})
    # heatmap of RMSE(gbdte)/RMSE(lgbm)
    sub = res[res["metric"] == "rmse"].groupby(["bench", "model"])["value"].mean().unstack()
    if {"gbdte", "lgbm"} <= set(sub.columns):
        ratio = np.full((len(RHO_GRID), len(DRIFT_GRID)), np.nan)
        for i, rho in enumerate(RHO_GRID):
            for j, dr in enumerate(DRIFT_GRID):
                name = f"rho{rho}_drift{dr}"
                if name in sub.index:
                    ratio[i, j] = sub.loc[name, "gbdte"] / sub.loc[name, "lgbm"]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(np.log2(ratio), cmap="RdBu", vmin=-2, vmax=2, origin="lower",
                       aspect="auto")
        ax.set_xticks(range(len(DRIFT_GRID)), [str(d) for d in DRIFT_GRID])
        ax.set_yticks(range(len(RHO_GRID)), [str(r) for r in RHO_GRID])
        ax.set_xlabel("drift strength")
        ax.set_ylabel(r"role purity $\rho$")
        fig.colorbar(im, ax=ax, label=r"$\log_2$ RMSE(GBDTE)/RMSE(LightGBM)")
        fig.tight_layout()
        fig.savefig(out / "regime_map.pdf")
        plt.close(fig)


def suite_rolestat_validation(out: Path, seeds: int, quick: bool) -> None:
    base = PRESETS["regime_base"]
    rows = []
    for rho in RHO_GRID:
        for seed in _seeds(seeds):
            cfg = replace(base, rho=rho, seed=seed, n_rows=400 if quick else base.n_rows)
            rows.append(dict(rho=rho, seed=seed,
                             separability=separability_index(generate(cfg))))
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps(
        {"suite": "rolestat_validation", "git": _git_sha(), "seeds": seeds}, indent=2))
    agg = df.groupby("rho")["separability"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker="o", capsize=3)
    ax.set_xlabel(r"role purity $\rho$")
    ax.set_ylabel("separability index")
    fig.tight_layout()
    fig.savefig(out / "separability_vs_rho.pdf")
    plt.close(fig)


def suite_auto_roles(out: Path, seeds: int, quick: bool) -> None:
    res = _regime_results(seeds, quick,
                          ["gbdte", "gbdte_auto", "gbdte_wrong", "gbdte_const"])
    _write_report(out, res, {"suite": "auto_roles", "git": _git_sha(), "seeds": seeds},
                  {"rmse": False})


SUITES = {
    "baselines_mse": suite_baselines_mse,
    "baselines_logloss": suite_baselines_logloss,
    "regime_map": suite_regime_map,
    "rolestat_validation": suite_rolestat_validation,
    "auto_roles": suite_auto_roles,
}


def run_suite(name: str, out_root: Path, seeds: int = 10, quick: bool = False) -> Path:
    out = Path(out_root) / name
    SUITES[name](out, seeds, quick)
    return out


__all__ = ["SUITES", "run_suite", "RHO_GRID", "DRIFT_GRID"]
```

`scripts/run_article_experiments.py`:

```python
#!/usr/bin/env python3
"""Run article experiment suites. Usage:
PYTHONPATH=python .venv/bin/python scripts/run_article_experiments.py --suite all --seeds 10
"""
import argparse
from pathlib import Path

from extra_boost_py.experiments.suites import SUITES, run_suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["all", *SUITES])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-scale run")
    ap.add_argument("--out", default="reports/article_experiments")
    args = ap.parse_args()

    names = list(SUITES) if args.suite == "all" else [args.suite]
    for name in names:
        out = run_suite(name, Path(args.out), seeds=args.seeds, quick=args.quick)
        print(f"[done] {name} -> {out}")


if __name__ == "__main__":
    main()
```

`python/tests/test_suites.py`:

```python
from pathlib import Path
from extra_boost_py.experiments.suites import run_suite


def test_quick_regime_map(tmp_path: Path):
    out = run_suite("regime_map", tmp_path, seeds=2, quick=True)
    assert (out / "results.csv").exists()
    assert (out / "run_meta.json").exists()


def test_quick_rolestat(tmp_path: Path):
    out = run_suite("rolestat_validation", tmp_path, seeds=2, quick=True)
    assert (out / "separability_vs_rho.pdf").exists()
```

- [ ] **Step 1: Write tests, verify FAIL; implement suites.py + CLI; verify PASS**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests/test_suites.py -q
```

- [ ] **Step 2: Full test suite green**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests -q
```

- [ ] **Step 3: Commit**

```bash
git add python/extra_boost_py/experiments/suites.py scripts/run_article_experiments.py python/tests/test_suites.py
git commit -m "Add experiment suites and CLI (E2 regime map, reports)"
```

---

### Task 7: Full runs + verification + snapshot refresh

**Files:**
- Create: `reports/article_experiments/*` (generated)
- Modify: `~/prj/gbdte_article_2026/` — copy reports, update `paper_progress.md` evidence section

- [ ] **Step 1: Quick end-to-end sanity**

```bash
PYTHONPATH=python .venv/bin/python scripts/run_article_experiments.py --suite all --seeds 2 --quick
```
Expected: five `[done] <suite>` lines; inspect one summary md.

- [ ] **Step 2: Full runs** (background; hours-scale is acceptable, monitor)

```bash
PYTHONPATH=python .venv/bin/python scripts/run_article_experiments.py --suite all --seeds 10
```

- [ ] **Step 3: Verify outputs** — every suite dir has results.csv, run_meta.json; baselines
  suites have cd_*.pdf and tables; regime_map.pdf exists; read the summary tables and check
  they tell a coherent story (gbdte best at rho=1/high drift; loses or ties at rho=0/drift=0;
  gbdte_auto ≈ gbdte; gbdte_wrong/all_in_leaf degraded; oracle best on logloss).

- [ ] **Step 4: Commit reports** (small text/pdf artifacts only) and copy to article repo

```bash
git add reports/article_experiments
git commit -m "Add article experiment reports (baselines, regime map, rolestat, auto roles)"
cp -r reports/article_experiments ~/prj/gbdte_article_2026/gbdte-main/reports/
```

- [ ] **Step 5: Update `~/prj/gbdte_article_2026/paper_progress.md`** — mark evidence rows
  (results tables, CD diagram, regime map, separability statistic) as available with paths.

---

## Self-Review Notes

- Spec coverage: E1→Task1, E7→Task4, E4/E5→Task3(+auto model in Task4), E6→Task5, E2→Task6 regime_map, oracle→Task2; E3 stretch intentionally has no task (spec marks it optional; add later if time remains).
- Type consistency: `Bench`/`BenchConfig` defined in Task 1 and consumed by name everywhere; `Model.param_grid`/`available`/`fit`/`predict` uniform.
- Known risk: engine training time for 10 seeds × tuning on k128 — if too slow, reduce `tune_trials` for GBDTE or preset n_rows; record any such deviation in run_meta.json.

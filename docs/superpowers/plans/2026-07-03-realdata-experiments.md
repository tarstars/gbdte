# E3 Real-Data Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `realdata` suite: Weather + Sberbank + HomeCredit with temporal splits, GBDTE (auto roles) vs external boosters, separability statistic computed pre-training, standard 5-seed reports plus a gain-vs-separability figure.

**Architecture:** One new module `realdata.py` (registry + download/cache + loader → standard `Bench`) and one new suite in `suites.py`. Everything else (models, stats, CLI) is reused unchanged.

**Tech Stack:** existing venv + `kaggle` CLI (installed, creds verified), pandas/pyarrow for parquet.

## Global Constraints

- Raw data + caches in `datasets/realdata/` (add to `.gitignore`); loaders download on first use and cache a compact parquet.
- Seeds = row subsamples (default cap 60k, revisable by the timing probe; record in run_meta).
- Temporal split at the 0.75 time quantile; tuning validation = train's last-by-t quarter (existing `_val_split`).
- Download failure → dataset skipped with an actionable message, suite continues.
- Tests must pass without network/data (`pytest -q`); data-dependent tests skip if cache absent.

---

### Task 1: Loader transform (offline-testable core)

**Files:**
- Create: `python/extra_boost_py/experiments/realdata.py`
- Test: `python/tests/test_realdata.py`
- Modify: `.gitignore` (add `datasets/realdata/`)

**Interfaces:**
- Produces: `RealDatasetSpec` (frozen dataclass: name, kaggle_ref, kaggle_kind
  ("dataset"|"competition"), files: tuple[str, ...], time_col, target_col, task),
  `REAL_DATASETS: dict[str, RealDatasetSpec]` for weather/sberbank/homecredit,
  `frame_to_bench(df, spec, seed, n_max) -> Bench` (pure transform, no I/O),
  `load_real(name, seed=0, n_max=60000, cache_dir=Path("datasets/realdata")) -> Bench`.

- [ ] **Step 1: failing tests** (`python/tests/test_realdata.py`)

```python
import numpy as np
import pandas as pd
import pytest

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
```

- [ ] **Step 2: run, verify FAIL** — `PYTHONPATH=python .venv/bin/pytest python/tests/test_realdata.py -q`

- [ ] **Step 3: implement `realdata.py`**

```python
"""E3: real datasets with temporal splits, loaded into the standard Bench form.

Raw data comes from Kaggle (credentials required); loaders download on first use into
datasets/realdata/ (gitignored) and cache a compact parquet extract. The transform is
pure and unit-tested offline: numeric features kept, categoricals ordinal-encoded,
feature NaNs median/`-1` imputed (no target involved), time normalized to [0,1],
temporal split at the 0.75 quantile.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .benchgen import Bench


@dataclass(frozen=True)
class RealDatasetSpec:
    name: str
    kaggle_ref: str
    kaggle_kind: str          # "dataset" | "competition"
    files: Tuple[str, ...]
    time_col: str
    target_col: str
    task: str                 # "mse" | "logloss"


REAL_DATASETS = {
    "weather": RealDatasetSpec(
        name="weather", kaggle_ref="pcovkrd84mejm/tabred-weather",
        kaggle_kind="dataset", files=("weather.parquet",),
        time_col="fact_time", target_col="fact_temperature", task="mse"),
    "sberbank": RealDatasetSpec(
        name="sberbank", kaggle_ref="sberbank-russian-housing-market",
        kaggle_kind="competition", files=("train.csv.zip",),
        time_col="timestamp", target_col="price_doc", task="mse"),
    "homecredit": RealDatasetSpec(
        name="homecredit", kaggle_ref="home-credit-credit-risk-model-stability",
        kaggle_kind="competition",
        files=("csv_files/train/train_base.csv",
               "csv_files/train/train_static_0_0.csv",
               "csv_files/train/train_static_0_1.csv",
               "csv_files/train/train_static_cb_0.csv"),
        time_col="date_decision", target_col="target", task="logloss"),
}


def frame_to_bench(df: pd.DataFrame, spec: RealDatasetSpec, seed: int,
                   n_max: int = 60000) -> Bench:
    df = df.dropna(subset=[spec.time_col, spec.target_col])
    if len(df) > n_max:
        rng = np.random.default_rng(seed)
        df = df.iloc[rng.choice(len(df), n_max, replace=False)]
    t_raw = pd.to_datetime(df[spec.time_col]).astype("int64").to_numpy(dtype=np.float64)
    t = (t_raw - t_raw.min()) / max(t_raw.max() - t_raw.min(), 1.0)

    out = {"t": t, "e_0": np.ones(len(df)), "e_1": t,
           "y": df[spec.target_col].to_numpy(dtype=np.float64)}
    partition_cols = []
    for col in df.columns:
        if col in (spec.time_col, spec.target_col):
            continue
        s = df[col]
        if np.issubdtype(s.dtype, np.number):
            v = s.to_numpy(dtype=np.float64)
            med = np.nanmedian(v)
            out[col] = np.where(np.isnan(v), 0.0 if np.isnan(med) else med, v)
        else:
            codes = s.astype("category").cat.codes.to_numpy(dtype=np.float64)
            out[col] = codes  # NaN becomes -1: its own category
        partition_cols.append(col)

    frame = pd.DataFrame(out).sort_values("t").reset_index(drop=True)
    cut = float(frame["t"].quantile(0.75))
    return Bench(df=frame, partition_cols=partition_cols,
                 extra_cols=["e_0", "e_1"], task=spec.task, cut=cut)


def _download(spec: RealDatasetSpec, cache_dir: Path) -> Path:
    raw = cache_dir / spec.name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    kind = "datasets" if spec.kaggle_kind == "dataset" else "competitions"
    for f in spec.files:
        target = raw / Path(f).name
        if target.exists() or target.with_suffix("").exists():
            continue
        cmd = [".venv/bin/kaggle", kind, "download", spec.kaggle_ref,
               "-f", f, "-p", str(raw)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"kaggle download failed for {spec.name}: {res.stderr.strip()}\n"
                f"run manually: {' '.join(cmd)}")
    # unzip any zips kaggle left behind
    for z in raw.glob("*.zip"):
        subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(raw)], check=True)
        z.unlink()
    return raw


def _read_raw(spec: RealDatasetSpec, raw: Path) -> pd.DataFrame:
    if spec.name == "weather":
        return pd.read_parquet(raw / "weather.parquet")
    if spec.name == "sberbank":
        return pd.read_csv(raw / "train.csv")
    if spec.name == "homecredit":
        base = pd.read_csv(raw / "train_base.csv")
        for name in ("train_static_0_0.csv", "train_static_0_1.csv",
                     "train_static_cb_0.csv"):
            part = pd.read_csv(raw / name)
            part = part.drop_duplicates("case_id")
            base = base.merge(part, on="case_id", how="left",
                              suffixes=("", f"_{name.split('.')[0]}"))
        return base.drop(columns=["case_id", "MONTH", "WEEK_NUM"], errors="ignore")
    raise KeyError(spec.name)


def load_real(name: str, seed: int = 0, n_max: int = 60000,
              cache_dir: Path = Path("datasets/realdata")) -> Bench:
    spec = REAL_DATASETS[name]
    cache = Path(cache_dir) / name / "extract.parquet"
    if not cache.exists():
        raw = _download(spec, Path(cache_dir))
        df = _read_raw(spec, raw)
        # compact cache: pre-subsample very large frames to 400k rows deterministically
        if len(df) > 400000:
            df = df.iloc[np.random.default_rng(0).choice(len(df), 400000, replace=False)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    df = pd.read_parquet(cache)
    return frame_to_bench(df, spec, seed=seed, n_max=n_max)


__all__ = ["RealDatasetSpec", "REAL_DATASETS", "frame_to_bench", "load_real"]
```

- [ ] **Step 4: run, verify PASS; add `datasets/realdata/` to `.gitignore`; commit**

```bash
printf "datasets/realdata/\n" >> .gitignore
git add python/extra_boost_py/experiments/realdata.py python/tests/test_realdata.py .gitignore
git commit -m "Add real-data loaders with offline-tested transform (E3)"
```

---

### Task 2: Downloads, column verification, timing probe

**Files:** none created (cache only); possibly adjust `REAL_DATASETS` column names.

- [ ] **Step 1: verify column assumptions from tiny previews before big downloads**

```bash
.venv/bin/kaggle datasets download pcovkrd84mejm/tabred-weather -f weather_preview.csv -p /tmp/probe
cd /tmp/probe && unzip -o -q weather_preview.csv.zip 2>/dev/null; head -1 weather_preview.csv | tr ',' '\n' | head
```
Confirm `fact_time` and `fact_temperature` exist (TabReD datasheet names); likewise
`timestamp`/`price_doc` for sberbank (data dictionary) and `date_decision`/`target` in
`train_base.csv` (download that 100-MB-scale file directly). Fix `REAL_DATASETS` if names differ.

- [ ] **Step 2: materialize caches one by one** (background for weather, 3.6 GB)

```bash
PYTHONPATH=python .venv/bin/python -c "
from extra_boost_py.experiments.realdata import load_real
for name in ['sberbank', 'homecredit', 'weather']:
    b = load_real(name, seed=0)
    print(name, len(b.df), len(b.partition_cols), b.task, 'train', len(b.train))"
```

- [ ] **Step 3: timing probe** — one GBDTE auto-roles fit per dataset at 60k rows; if a
  fit exceeds ~300 s, lower `n_max` for that dataset in the suite (record the choice).

```bash
PYTHONPATH=python .venv/bin/python - <<'EOF'
import time
from extra_boost_py.experiments.realdata import load_real
from extra_boost_py.experiments.baselines import GBDTEModel
for name in ["sberbank", "homecredit", "weather"]:
    b = load_real(name, seed=0)
    m = GBDTEModel("auto")
    t0 = time.time(); m.fit(b, {"n_stages": 48, "max_depth": 5}); dt = time.time()-t0
    print(f"{name}: fit {dt:.0f}s")
EOF
```

- [ ] **Step 4: commit any registry fixes**

---

### Task 3: `realdata` suite

**Files:**
- Modify: `python/extra_boost_py/experiments/suites.py`
- Test: `python/tests/test_suites.py` (quick test, skipped without cache)

**Interfaces:** `suite_realdata(out, seeds, quick)` registered as `SUITES["realdata"]`.

- [ ] **Step 1: failing test**

```python
import pytest
from pathlib import Path

def test_quick_realdata(tmp_path: Path):
    if not Path("datasets/realdata/sberbank/extract.parquet").exists():
        pytest.skip("real data cache absent")
    out = run_suite("realdata", tmp_path, seeds=2, quick=True)
    assert (out / "results.csv").exists()
    assert (out / "separability.csv").exists()
```

- [ ] **Step 2: implement** (in `suites.py`; imports: `load_real`, `REAL_DATASETS`,
  `separability_index`, `drift_score`)

```python
def suite_realdata(out: Path, seeds: int, quick: bool) -> None:
    names = ["sberbank"] if quick else ["sberbank", "homecredit", "weather"]
    n_max = 2000 if quick else 60000
    model_names_mse = ["gbdte", "gbdte_auto", "gbdte_const", "lgbm", "lgbm_linear",
                       "xgb", "catboost"]
    sep_rows, all_results = [], []
    for name in names:
        task = REAL_DATASETS[name].task
        all_models = make_models(task, include_oracle=False)
        models = {k: v for k, v in all_models.items() if k in model_names_mse}

        def factory(bname: str, seed: int):
            bench = load_real(bname, seed=seed, n_max=n_max)
            sep_rows.append(dict(bench=bname, seed=seed,
                                 separability=separability_index(bench)))
            return bench

        res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                       bench_names=[name], tune_trials=2 if quick else 8)
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    hib = {"rmse": False, "auc": True, "logloss": False}
    _write_report(out, results, {"suite": "realdata", "git": _git_sha(),
                                 "seeds": seeds, "n_max": n_max}, hib)
    sep = pd.DataFrame(sep_rows).drop_duplicates(["bench", "seed"])
    sep.to_csv(out / "separability.csv", index=False)

    # gain vs separability: relative test metric GBDTE(auto)/LightGBM per dataset
    rows = []
    for name in names:
        metric = "rmse" if REAL_DATASETS[name].task == "mse" else "logloss"
        sub = results[(results["bench"] == name) & (results["metric"] == metric)]
        perf = sub.groupby("model")["value"].mean()
        if {"gbdte_auto", "lgbm"} <= set(perf.index):
            rows.append(dict(bench=name, metric=metric,
                             rel=perf["gbdte_auto"] / perf["lgbm"],
                             separability=sep[sep["bench"] == name]["separability"].mean()))
    gain = pd.DataFrame(rows)
    gain.to_csv(out / "gain_vs_separability.csv", index=False)
    if len(gain) >= 2:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.scatter(gain["separability"], gain["rel"])
        for _, r in gain.iterrows():
            ax.annotate(f' {r["bench"]}', (r["separability"], r["rel"]), fontsize=8)
        ax.axhline(1.0, color="0.6", lw=1, ls="--")
        ax.set_xlabel("separability index (pre-training)")
        ax.set_ylabel("GBDTE / LightGBM (lower = GBDTE wins)")
        fig.tight_layout()
        fig.savefig(out / "gain_vs_separability.pdf")
        plt.close(fig)
```

Register `SUITES["realdata"] = suite_realdata`.

- [ ] **Step 3: quick suite run + full pytest green; commit**

```bash
PYTHONPATH=python .venv/bin/pytest python/tests -q
git add python/extra_boost_py/experiments/suites.py python/tests/test_suites.py
git commit -m "Add realdata suite with pre-training separability reporting (E3)"
```

---

### Task 4: Full run, verification, notes

- [ ] **Step 1:** `--suite realdata --seeds 5` in background; monitor.
- [ ] **Step 2:** verify report coherence; check the separability→gain relation and note
  it honestly either way.
- [ ] **Step 3:** commit reports; update `~/prj/gbdte_article_2026/paper_progress.md`
  (E3 evidence section) per the notes rule.
- [ ] **Step 4:** finishing-a-development-branch.

## Self-Review Notes

- Spec coverage: loaders→T1, downloads/columns/timing→T2, suite+statistic+figure→T3,
  run+notes→T4. Auto-roles works via existing code (spec §5) — no code change needed.
- Column names for weather (`fact_time`, `fact_temperature`) are TabReD-datasheet-derived
  and explicitly verified in T2 Step 1 before anything depends on them.
- Type consistency: `frame_to_bench` returns the standard `Bench`; suite reuses
  `make_models`/`run_grid`/`_write_report` unchanged.

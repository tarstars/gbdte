# GBDTE (Extra Bridged Boosting)

Research-oriented implementation of ExtraBoost-style gradient boosting with a Go core and a Python bridge for experiments, metrics, and reports.

## What is it?
GBDTE is a research-grade implementation of ExtraBoost-style gradient boosting focused on extrapolating structured signals.
It couples a Go core learner with a thin Python bridge for dataset generation, training, and evaluation.
It ships end-to-end scripts and reports to reproduce the classical MSE/logloss experiments.

## Why it matters?
It targets settings where generalization beyond the training support matters, not just in-sample fit.
The repo exposes both code and experiment outputs so results can be audited and extended.
It is designed to stay lightweight for iterative research while remaining reproducible.

## Repository Layout
- `golang/extra_boost/ebl`: gradient boosting engine (tree growing, split search, losses) with Go unit tests.
- `golang/extra_boost/pybridge`: CGO entry point compiled into a shared library consumed from Python (`libextra_boost.*`).
- `golang/extra_boost/extra_boost_main`: minimal CLI harness for ad-hoc experimentation.
- `golang/poisson_legacy`: legacy Poisson booster and its CGO bridge for back-compat experiments.
- `python/extra_boost_py`: Python package providing ctypes bindings (`bridge.py`), high-level booster API, classical dataset generator, metrics, and the orchestration pipeline.
- `python/examples/full_pipeline.py`: end-to-end demo (generate dataset → train booster → evaluate → persist model).
- `python/report/generate_test_binary_report.py`: reusable reporter that produces uplift/ROC plots and Markdown summaries.
- `scripts/run_classical_experiments.py`: batch runner that regenerates the MSE and logloss experiment artefacts under `reports/classical_experiment/`.
- `scripts/run_smoke_tests.py`: quick smoke tests for the standard and legacy bridges.
- `reports/classical_experiment/`: latest experiment outputs (PDF charts, Markdown, JSON summaries).
- `docs/tech_report/`: outline for the technical report and citation pointers.

## Getting Started
Prerequisites: Go ≥ 1.19 with CGO enabled, Python ≥ 3.10 with NumPy and matplotlib available.

1. **Build the shared library** (needed once per environment):
   ```bash
   PYTHONPATH=python python3 -c "from extra_boost_py.go_lib import build_shared; build_shared()"
   ```
   This runs `go build -buildmode=c-shared` on `golang/extra_boost/pybridge` and drops `libextra_boost.*` into `python/extra_boost_py/`.

2. **Smoke-test the pipeline demo**:
   ```bash
   PYTHONPATH=python python3 python/examples/full_pipeline.py
   ```
   The script rebuilds the shared library if absent, trains on the classical dataset, evaluates ROC AUC over time, and saves model artefacts to `artifacts/`.

2b. **Smoke-test both standard and legacy Poisson bridges**:
   ```bash
   PYTHONPATH=python python3 scripts/run_smoke_tests.py
   ```
   This builds both shared libraries (`libextra_boost.*` and `libextra_poisson_legacy.*`) and runs the standard pipeline plus the Poisson legacy quickcheck.

3. **Regenerate the classical experiment reports** (produces Markdown, PDF plots, JSON summaries):
   ```bash
   PYTHONPATH=python python3 scripts/run_classical_experiments.py
   ```
   Results are written to `reports/classical_experiment/{mse,logloss}/`.

## Results and Reproducing Figure 1
- Run the classical experiment script:
  ```bash
  PYTHONPATH=python python3 scripts/run_classical_experiments.py
  ```
- Outputs land in `reports/classical_experiment/` with PDF charts and summaries.
- `artifacts/` and `libextra_*` files are transient build/run outputs and are intentionally untracked.

## Developing
- Run Go unit tests:
  ```bash
  go test ./golang/extra_boost/...
  ```
- When touching the Go/Python interface, re-run `build_shared()` and the pipeline demo to confirm ABI compatibility.
- The Python package exports `ExtraBooster`, `BoosterParams`, `generate_classical_dataset`, and `run_classical_pipeline` for notebook-style experimentation:
  ```python
  from extra_boost_py import BoosterParams, ExtraBooster, generate_classical_dataset, train_test_split

  dataset = generate_classical_dataset(n_samples=20_000)
  train, test = train_test_split(dataset, ratio=0.8)
  params = BoosterParams(loss="logloss", n_stages=300)
  booster = ExtraBooster.train(train.features_inter, train.features_extra, train.target, params=params)
  preds = booster.predict(test.features_inter, test.features_extra)
  ```

Keep the repository lean by regenerating artefacts only when needed and ensuring transient outputs (`libextra_*`, `artifacts/`) stay untracked.

## About metadata (GitHub)
Suggested description: “GBDTE: Go + Python ExtraBoost-style gradient boosting for extrapolation-focused research.”
Suggested topics: `gbdt`, `gradient-boosting`, `decision-trees`, `extrapolation`, `python`, `go`, `machine-learning`, `research`.

## License
Licensed under the Apache License, Version 2.0. See the LICENSE file for details.

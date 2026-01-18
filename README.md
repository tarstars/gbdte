# GBDTE — Extra Bridged Boosting (ExtraBoost)

Gradient boosted decision trees with extrapolation via extra (bridged) features.

## Naming
GBDTE is the project and research direction.
Extra Bridged Boosting (ExtraBoost) is the implementation/algorithm nickname used in code and reports.

## What is it?
GBDTE implements Extra Bridged Boosting (ExtraBoost) with a Go core learner and Python orchestration.
It provides dataset generation, training, and evaluation pipelines for classical MSE/logloss experiments.
The bridge is designed for fast iteration while keeping experiments reproducible.

## Why it matters?
It targets extrapolation and distribution-shift evaluation, not just in-sample fit.
The repo ships scripts and reports so results can be audited and extended.
It stays lightweight for research while keeping commands deterministic.

## Repository Layout
- `datasets/`: classical datasets and static assets used in the MSE experiments.
- `docs/`: tech report outline, release checklist, and presentation materials.
- `golang/extra_boost/ebl`: gradient boosting engine (tree growing, split search, losses) with Go unit tests.
- `golang/extra_boost/pybridge`: CGO entry point compiled into a shared library consumed from Python (`libextra_boost.*`).
- `golang/extra_boost/extra_boost_main`: minimal CLI harness for ad-hoc experimentation.
- `golang/poisson_legacy`: legacy Poisson booster and its CGO bridge for back-compat experiments.
- `notebooks/`: experiment notebooks for MSE/logloss analyses (including archived variants).
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

## GitHub About suggestion
Suggested description: GBDTE: gradient boosted decision trees with extrapolation via extra (bridged) features.
Suggested topics: gbdt, gradient-boosting, decision-trees, extrapolation, python, go, machine-learning, research

## License
Licensed under the Apache License, Version 2.0. See the LICENSE file for details.

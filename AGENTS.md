# Repository Guidelines

## Project Structure & Module Organization
- `golang/extra_boost/ebl`: core gradient boosting engine in Go; unit tests live beside the implementation.
- `golang/extra_boost/pybridge`: CGO entry point compiled into the shared library consumed by Python.
- `golang/extra_boost/extra_boost_main`: minimal CLI harness for manual experiments.
- `python/extra_boost_py`: Python bindings, ctypes bridge, classical dataset generator, and pipeline utilities.
- `python/examples/full_pipeline.py`: end-to-end demo covering dataset generation, training, and evaluation.
- `scripts/run_classical_experiments.py`: reproducible experiment runner; `reports/` and `docs/` capture generated artefacts and references.

## Build, Test, and Development Commands
- `go build ./...`: compile every Go package (useful before publishing shared objects).
- `go test ./...`: execute Go unit tests located in `golang/extra_boost/ebl`.
- `PYTHONPATH=python python3 python/examples/full_pipeline.py`: smoke-test the bridge and pipeline using the demo workflow.
- `python3 -c "from extra_boost_py.go_lib import build_shared; build_shared()"`: build `libextra_boost.*` and its header via the helper.
- `PYTHONPATH=python python3 scripts/run_classical_experiments.py`: rerun the classical benchmarking experiments.

## Coding Style & Naming Conventions
- Go code must be formatted with `gofmt`; exported identifiers stay in CamelCase, while helpers remain unexported in lowerCamelCase.
- Go tests follow `TestSubjectScenario` naming and live alongside the code they verify.
- Python modules use 4-space indentation, snake_case functions, and type hints (see `BoosterParams` dataclass in `python/extra_boost_py/booster.py`).
- Validate array-oriented code with NumPy vectorisation; raise informative `ValueError` exceptions on shape mismatches, as the bridge already does.

## Testing Guidelines
- Add Go tests under `golang/extra_boost/ebl` with `_test.go` filenames and table-driven cases to cover numerical edge conditions.
- Whenever the Go/Python interface changes, rebuild the shared library and rerun the full pipeline demo to confirm integration still holds.
- For Python additions, include targeted doctests or lightweight assertions inside examples until a dedicated test suite is introduced.

## Commit & Pull Request Guidelines
- Prefer concise, present-tense subject lines (`Refine booster constructor`, `Fix split search bounds`); include scope prefixes when touching multiple layers.
- Reference the impacted package(s) in the body, list manual checks (`go test ./...`, pipeline demo), and link relevant experiment reports.
- Pull requests should describe motivation, outline validation steps, attach generated artefacts (graphs, metrics), and note any follow-up work required.

## Bridge Configuration Tips
- Ensure `CGO_ENABLED=1` (default in the helper) and a writable build directory before invoking `build_shared()`.
- Use the `EXTRA_BOOST_LIB` environment variable to point Python to a custom shared library location when testing alternate builds.

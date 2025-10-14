Python Wrapper Strategy for the Go Booster
=========================================

Goals
-----
- Offer a Jupyter-friendly Python package that wraps the Go-based booster (`golang/extra_boost/ebl`).
- Provide train / predict / render functionality directly on NumPy arrays, with minimal friction.
- Ship the wrapper as a distributable package (PyPI-ready), hiding the Go toolchain from end-users as much as possible.

What the Go Side Exposes
------------------------
- Core API lives in `golang/extra_boost/ebl`:
  * `NewEBooster(EBoosterParams)` trains a model given an `EMatrix` (inter features, extra features, target) and hyperparameters (`ebooster.go:55`).
  * `EBooster.PredictValue` performs inference on dense matrices (`ebooster.go:88`).
  * `EBooster.Save` / `LoadModel` persist models as JSON (`ebooster.go:107`, `ebooster.go:136`).
  * `EBooster.RenderTrees` and `DumpLearningCurves` are optional utilities.
- Datasets are represented as `EMatrix` objects containing `mat.Dense` values (`ematrix.go:10`).
- The CLI in `extra_boost_main` simply orchestrates JSON config parsing and file I/O around this library.

Integration Options Considered
------------------------------
1. **Subprocess CLI bridge** – call the existing Go binary via `subprocess` and exchange `.npy` files. Simple, but slow and awkward in notebooks.
2. **gRPC / HTTP server** – run the Go booster as a long-lived service. Adds deployment burden and networking overhead.
3. **Direct shared library binding** – compile Go as a C shared library (`-buildmode=c-shared`) and call it from Python with `ctypes`/`cffi`. Best balance for notebook and PyPI use.

Proposed Architecture
---------------------
### Build artefact
- Create a dedicated Go entry point (e.g. `golang/extra_boost/pybridge` package) that wraps the `ebl` API in C-friendly functions, exporting handles such as `TrainModel`, `Predict`, `SaveModel`, `LoadModel`, `FreeBuffer`.
- Compile this package into `libextra_boost.so` + header using `go build -buildmode=c-shared`.
- Capture the ABI as functions that accept pointers/lengths or JSON-serialized payloads:
  * Pass dense data as contiguous `float64*` with explicit `rows`, `cols`.
  * Return predictions via Go-allocated buffers, plus a release function the Python side can call.

### Python package layout (`extra_boost_py/`)
- `__init__.py` exports high-level classes (`ExtraBooster`, `EBoosterParams`).
- `_bridge.py` uses `ctypes` (or `cffi`) to load `libextra_boost.so`, map argument/return types, and provide minimal wrappers.
- `dataset.py` hosts helpers to map `numpy.ndarray` ↔ raw C pointers without copies (using `ndarray.ctypes.data`/`numpy.ascontiguousarray`).
- `training.py` exposes `train(features_inter, features_extra, target, params)` returning an `ExtraBooster` object with `.predict()`/`.save()`.

### Build & distribution
- Use `setuptools` `build_ext` subclass to invoke `go build` during wheel creation. For example:
  ```python
  class GoBuildExt(build_ext):
      def run(self):
          subprocess.check_call([
              "go", "build", "-o", os.path.join(self.build_temp, "libextra_boost.so"),
              "-buildmode=c-shared", "./golang/extra_boost/pybridge",
          ])
          copy the `.so` + generated header into the package.
  ```
- Provide prebuilt wheels for major OS/architectures when possible to avoid requiring Go on end-user machines. Otherwise, document Go (>=1.19) as a build dependency.
- Add `pyproject.toml` with a custom backend (or `setuptools.build_meta`) to ensure the Go build runs.

### Jupyter UX
- Present straightforward APIs:
  ```python
  from extra_boost_py import ExtraBooster, BoosterParams

  booster = ExtraBooster.train(features_inter=X, features_extra=XE, target=y,
                               params=BoosterParams(n_stages=200, max_depth=6, ...))
  preds = booster.predict(X_val, XE_val)
  booster.save("model.ebm")
  ```
- Support NumPy arrays directly; optionally accept Pandas DataFrames (convert internally).
- Provide plotting utilities that wrap `RenderTrees` output (e.g., call exported bridge to dump Graphviz `.dot` or use Python’s `graphviz` to visualize returned structures).

Implementation Steps
--------------------
1. **Design C ABI:** add a Go `pybridge` package with exported functions (using `//export` comments) that:
   - Accept/return raw pointers for matrices and predictions.
   - Marshal hyperparameters via JSON or struct arguments.
   - Manage Go memory life cycle (explicit `Free` function).
2. **Create Python bridge module:** load the shared library with `ctypes`, map argument types, provide thin wrappers that convert NumPy arrays and handle ownership.
3. **High-level Python API:** implement ergonomic classes/functions, mirroring the Go semantics but using Pythonic defaults and dataclasses for params.
4. **Packaging tooling:** write `setup.py` / `pyproject.toml` to run `go build`, include `.so` in wheels, and add metadata (trove classifiers, etc.).
5. **Documentation & tests:** add Jupyter notebooks demonstrating training and prediction; write PyTest-based integration tests that train on small synthetic data and verify parity with Go CLI results.

Open Questions / Risks
----------------------
- Need to ensure Go’s `mat.Dense` expects column-major data; confirm Python arrays are converted appropriately to avoid copying/misalignment.
- Decide on model serialization: re-use Go’s `Save`/`Load` JSON to keep compatibility, or implement direct memory snapshot.
- Plan for thread-safety—exported Go functions may not be safe for concurrent use without additional locking.


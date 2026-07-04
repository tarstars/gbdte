# Poisson Mode

GBDTE's third loss, implemented by the legacy engine in `golang/poisson_legacy` and
exposed in Python as `extra_boost_py.PoissonLegacyBooster`. This page is the
GitHub-friendly summary; the full mathematical write-up with references is in
[`poisson_mode_explained.pdf`](poisson_mode_explained.pdf), and a runnable walkthrough
is in [`../../notebooks/poisson_mode/poisson_mode_demo.ipynb`](../../notebooks/poisson_mode/poisson_mode_demo.ipynb).

## What it models

Count data with a **time-varying event rate**. Where LightGBM/XGBoost/CatBoost's
Poisson objectives predict one expected count per feature vector, GBDTE models an
*intensity* $\lambda(t \mid x)$ — events per unit time — and extrapolates its trend
into a future window:

- **partition features** (stable object attributes) drive the tree splits and select
  the intensity regime;
- each leaf stores a weight vector $w$ over an **extrapolation basis** $\varphi(t)$
  (e.g. $(1, t)$) and predicts $\lambda(t) = w^\top \varphi(t)$ — a linear intensity
  per leaf, so the fitted rate carries its trend beyond the training time range.

Typical use cases: insurance claim frequency, failure rates, call/traffic volumes —
anywhere counts drift over time and the future is what matters. Formally this is the
inhomogeneous Poisson process setting; see the PDF for the likelihood and literature.

## Data contract

One row is **(object, time-bin, count)**: an object observed over the window
$[0, c]$ split into bins contributes one row per bin.

| Argument         | Shape        | Meaning                                                        |
|------------------|--------------|----------------------------------------------------------------|
| `bjids`          | `(n,)` int32 | object (cohort) id; rows of one object share it                 |
| `freqs`          | `(n,)` f64   | event **count** in the row's bin; **must be > 0**               |
| `features_inter` | `(n, p)` f64 | partition features (constant within an object)                  |
| `features_extra` | `(n, d)` f64 | $\varphi(t)$ evaluated at the row's bin center                  |
| `psi`            | `(d,)` f64   | $\int_0^c \varphi(t)\,dt$ over the whole observation window     |

For the basis $(1, t)$ and window $[0, c]$: `psi = [c, c**2 / 2]`.

`predict(features_inter, features_extra)` returns the expected **per-bin count**
(intensity in count units), one value per row.

## Quick start

```python
import numpy as np
from extra_boost_py import PoissonLegacyBooster, PoissonLegacyParams

# rows: (object, bin) with counts; two groups with rates 2+2t and 4-2t
rng = np.random.default_rng(0)
n_obj, n_bins, exposure = 200, 10, 20
edges = np.linspace(0.0, 1.0, n_bins + 1)
centers, delta = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]

rows = []
for j in range(n_obj):
    g = j % 2
    lam = 2 + 2 * centers if g == 0 else 4 - 2 * centers
    counts = np.maximum(rng.poisson(lam * delta * exposure), 1)   # counts must be > 0
    rows += [(j, counts[b], float(g), centers[b]) for b in range(n_bins)]
arr = np.array(rows)

booster = PoissonLegacyBooster.train(
    bjids=arr[:, 0].astype(np.int32),
    freqs=arr[:, 1],
    features_inter=arr[:, 2].reshape(-1, 1),
    features_extra=np.column_stack([np.ones(len(arr)), arr[:, 3]]),  # phi(t) = (1, t)
    psi=np.array([1.0, 0.5]),                                        # integral over [0, 1]
    params=PoissonLegacyParams(n_stages=1, max_depth=3, learning_rate=1.0),
)
# predicted per-bin counts at t = 0.9 for group 0; divide by exposure*delta for the unit rate
pred = booster.predict(np.array([[0.0]]), np.array([[1.0, 0.9]]))
print(pred / (exposure * delta))   # ~ 2 + 2*0.9 = 3.8
```

Build the shared library first if you haven't:
`python -c "from extra_boost_py.go_lib import build_shared; build_shared()"` builds the
main engine; the Poisson legacy library builds via
`scripts/run_smoke_tests.py` or `build_poisson_legacy_shared()` from the same module.

## Status and limitations

Behaviour is pinned by `python/tests/test_poisson_semantics.py`.

1. **Multi-stage boosting: FIXED (2026-07-04).** Earlier only `n_stages=1` was usable —
   every later stage added a systematic offset because the Newton exposure term used
   `N_L·psi` (per-object integral of phi) instead of the per-row running sum `sum_i phi_i`,
   a bin-width factor too small. The engine now uses the per-row exposure; multi-stage
   converges to the true intensity (test checks `n_stages` in {2, 5}). Analysis:
   [`poisson_mode_explained.pdf`](poisson_mode_explained.pdf) §4–5.
2. **Zero counts crash the process** (an abort inside the CGO layer, not a Python
   exception). Aggregate to coarser bins or larger cohorts so counts stay ≥ 1.
3. **Leaf-intensity stability across depth (open).** The additive linear leaf can dip
   toward zero and inflate the Poisson deviance, so results are sensitive to tree depth;
   the experiment wrapper applies an intensity floor (1% of the mean count) as a partial
   mitigation. A positivity-constrained leaf solve in the engine is the proper fix
   (future work). At a stable shallow depth, GBDTE-Poisson is competitive with LightGBM
   on the synthetic benchmark.

## Benchmark and experiments

The synthetic benchmark ("grouped drifting intensity") lives in
`python/extra_boost_py/experiments/poisson_bench.py`: latent groups with linear
per-unit intensities, cohort exposure, role-purity and drift knobs, a $t < 0.6$
temporal train/test split, and a `lam_true` column enabling an exact oracle. The
baseline matrix (GBDTE role variants, LightGBM/XGBoost/CatBoost Poisson objectives,
Bayes oracle) with 10-seed statistics runs via:

```bash
PYTHONPATH=python .venv/bin/python scripts/run_article_experiments.py \
    --suite baselines_poisson --seeds 10
```

Reports land in `reports/article_experiments/baselines_poisson/`.

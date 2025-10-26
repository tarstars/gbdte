# Extra Boosting (MSE) Dataset — v1

A compact, reproducible dataset for testing models that **greedily decompose** a 1‑D signal into components using the basis
\[1, *t*, sin(*k t*), cos(*k t*)\] under a **mean squared error (MSE)** objective.

This release accompanies the “Extra Boost” dot‑trace experiment and is designed for training/evaluating models that
select components and fit linear combinations of the given basis functions.

---

## What’s in the dataset

### Files
- `extra_boosting_static_features_k50.csv` — the **training table** (static features). Columns:
  - `f_1, f_2, f_3` — 3‑bit binary encoding of the component id: encode `(component_iter − 1)` with `f_1` as MSB and `f_3` as LSB.
    - Iteration 1 → `000`, 2 → `001`, …, 8 → `111`.
  - `e_1, e_2, e_3, e_4` — static features derived from `t`:
    - `e_1 = 1`
    - `e_2 = t`
    - `e_3 = sin(k t)`
    - `e_4 = cos(k t)`
  - `t` — horizontal coordinate in \[0, 1\].
  - `y` — target constructed from the component’s linear combination:  
    \[\; y = A + B\,t + C\,\sin(k\,t) + D\,\cos(k\,t) \;\] with **fixed** `k = 50`.

- `extra_boosting_dataset_k50.csv` — (optional diagnostic table, if present) includes extra columns:
  - `component_iter, k, tol, A, B, C, D, y_obs` alongside `t, y, e_1..e_4`.

> Total rows in this release (k = 50, tol = 0.01, 8 greedy iterations): **244**  
> Per‑iteration captures: **51, 39, 33, 32, 26, 24, 22, 17**.

---

## How the data were created

1. **Dot extraction** from a photo of the hand‑written “Extra Boost” title made of dots.  
   - Convert to grayscale → Otsu threshold (with a small bias) → connected components → keep centroids with area in a safe range.
   - Produce **RAW coordinates** with `x` normalized by image width and `y` scaled by width, origin at the **bottom‑left**.

2. **Signal window** in RAW space: keep only points with `0.2 < y < 0.6` (removes shadows/background speckles).

3. **Normalization** (uniform): rescale so the **horizontal span** equals `1.0` and shift so the **left‑bottom** point is `(0,0)`.
   Use the normalized `x` as `t`.

4. **Greedy decomposition** (maximum consensus / RANSAC‑style), **fixed frequency** `k = 50` and tolerance `±0.01`:
   - Model: \( y = A + B t + C \sin(k t) + D \cos(k t) \).
   - At each iteration, sample minimal sets, fit \(A,B,C,D\), keep the parameters that **maximize inliers** (`|y − ŷ| ≤ tol`),
     then **refit on inliers** and remove them. Repeat for 8 iterations.

5. **Static features**: for every captured point, compute
   - component bits `f_1..f_3` (binary encoding of the component id),
   - `e_1..e_4 = (1, t, sin(k t), cos(k t))`,
   - target `y` as the model value at the point’s `t`.

---

## Intended use

- Train/evaluate models that learn to **reconstruct** a signal from static features and component indicators under **MSE**.
- Benchmark greedy / boosting‑style selection strategies with a fixed basis.

This dataset is **1‑D** and intentionally compact to simplify rapid experimentation and visualization.

---

## Quick start (Python)

```python
import pandas as pd
import numpy as np

df = pd.read_csv("extra_boosting_static_features_k50.csv")

X = df[["f_1","f_2","f_3","e_1","e_2","e_3","e_4","t"]].values
y = df["y"].values

# Example: fit a linear model
from sklearn.linear_model import Ridge
m = Ridge(alpha=1e-6).fit(X, y)
print("R^2:", m.score(X, y))

# Or XGBoost / LightGBM:
# import xgboost as xgb
# m = xgb.XGBRegressor(max_depth=3, n_estimators=200, learning_rate=0.05).fit(X, y)
```

---

## Reproducibility

You can regenerate the dataset from the source photo with the helper script:

```bash
python make_extra_boosting_dataset.py   --image path/to/photo.png   --k 50 --iters 8 --tol 0.01   --ymin 0.2 --ymax 0.6   --outdir out
```

This will write:
- `out/points_raw.csv`, `out/signal_normalized.csv`
- per‑iteration reports and masks: `out/iter_XX_report.json`, `out/iter_XX_points.csv`
- the final table: `out/extra_boosting_static_features.csv`

---

## Notes & caveats

- The **binary component flags** (`f_1..f_3`) encode only which greedy component a point belongs to; they don’t impose order beyond the 3‑bit code.
- `y` is **synthetic**, computed from the fitted component at `t`. If you also want the **observed** values, use the diagnostic table (`y_obs`). 
- The decomposition depends on `k` and `tol`. Changing either will change captures and counts.

---

## Citation / attribution

If you use this dataset, please link back to the “Extra Boosting (MSE) Dataset — v1” post and the GitHub repository.
### Filtered component dataset
- `component_f000.csv` — subset containing only rows with `(f_1, f_2, f_3) = (0, 0, 0)`; useful for verifying leaf behaviour and linear fits in isolation.


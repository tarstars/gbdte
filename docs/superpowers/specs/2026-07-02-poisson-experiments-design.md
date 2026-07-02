# Poisson Mode Experiments Design

Date: 2026-07-02
Status: approved-by-default (user AFK at question time; the recommended option was taken and
every decision below is open to revision)
Extends: `2026-07-02-article-experiments-design.md` (E1–E7 layer, now on `main`)

## Goal

Give the paper's third loss (Poisson — named in the presentation as a GBDTE differentiator:
"LightGBM: the closest match (MSE, Logloss, Poisson); no extra/interpolation division") the
same reviewer-proof treatment as MSE and LogLoss: a principled dataset generator, a baseline
matrix with external Poisson objectives, ablations, 10-seed statistics with CD diagram.

## Decisions taken

1. **Data model: grouped drifting intensity** — the exact analog of the paper's other
   benchmarks. Objects have binary partition features encoding a latent group $g$ of $k$
   (plus the continuous nuisance `f_m`); each group has event intensity
   $\lambda_g(t) = \beta_g^\top \varphi(t)$ with $\varphi(t) = (1, t)$, kept positive by
   construction. `drift` scales the time component of $\beta$; `rho` mixes in an
   $m$-dependent $\tilde\beta$ exactly as in `benchgen`.
2. **Engine**: the existing `PoissonLegacyBooster` (Go, CGO). No engine changes.
3. **Empirical semantics validation comes first**: a test fits a known two-group linear
   intensity and asserts the engine recovers it. If this fails, implementation stops and the
   findings are reported — nothing else is built on guessed semantics.
4. Suite scope: one new suite `baselines_poisson` (matrix + ablations + stats). No Poisson
   regime map for now (YAGNI; the MSE regime map already carries that argument).

## Generator: `poisson_bench.py`

`PoissonBenchConfig`: k_groups=8, rho=1.0, drift=1.0, n_objects=500, n_bins=20, cut=0.6,
base_rate range (1..5 events per unit time), seed.

Construction:
- per object: group bits (as in `benchgen`), nuisance $m \sim U(0,1)$;
- $\beta_{g,0} \sim U(1,5)$, $\beta_{g,1} = drift \cdot u \cdot \beta_{g,0}$ with
  $u \sim U(-0.8, 0.8)$ so that $\lambda_g(t) > 0$ on $[0,1]$;
- impure part: $\tilde\beta_0(m) = 3 + 2\sin(2\pi m)$, $\tilde\beta_1(m) = drift \cdot 2\cos(2\pi m)$,
  clipped so the mixed intensity stays $\ge 0.1$;
- $\lambda(t) = \max(\rho\,\lambda_{group}(t) + (1-\rho)\,\lambda_m(t),\ 0.05)$;
- time binned into `n_bins` equal bins on $[0,1]$; per object×bin count
  $y \sim \mathrm{Pois}(\lambda(t_b)\,\Delta)$ where $t_b$ is the bin center.

`PoissonBench` (dataclass) exposes:
- `df`: tabular form — rows = object×bin with columns `f_*`, `f_m`, `t` (bin center),
  `y` (count), `bjid`, `lam_true` (true intensity), `group`; `train`/`test` properties split
  by `t` vs `cut`;
- `event_train()`: the GBDTE form — `(bjids, freqs, features_inter, features_extra, psi)`
  restricted to the train window, with `psi = ∫_0^{cut} φ(t) dt` computed analytically
  = `(cut, cut²/2)`;
- `delta`: bin width, for rate↔count conversion;
- `partition_cols`, `extra_cols` as in `Bench`.

## Models: extend `baselines.py`

- `GBDTEPoissonModel(roles="oracle" | "wrong" | "const")`:
  - oracle: inter = partition cols, extra = φ(t) with psi;
  - wrong: inter = `[t]`, extra = `(1, mean partition signal)` is meaningless for an
    integral-based likelihood, so "wrong" instead swaps by using `f_m` + `t` as inter and
    dropping the partition bits (mis-specified partitioning) — records the degradation;
  - const: basic mode (no features_extra) — constant per-leaf rate, the "no extrapolation"
    ablation;
  - `predict(df)` returns expected count per row: engine rate output × calibration factor
    settled by the semantics-validation test (documented in code).
- `LightGBMPoisson`, `XGBoostPoisson`, `CatBoostPoisson`: native `objective="poisson"`
  (`count:poisson` for xgb, `Poisson` for catboost) on `partition_cols + ["t"]`, target =
  per-bin count. Same param grids as their siblings.
- `PoissonOracleModel`: predicts `lam_true * delta`.

## Metrics: extend `stats.py`

`evaluate` gains task "poisson":
- `poisson_dev`: mean Poisson deviance
  $2\,[y \log(y/\hat\mu) - (y - \hat\mu)]$ (terms with $y{=}0$ handled as $2\hat\mu$),
  computed on future-window counts, lower is better (primary);
- `rate_rmse`: RMSE between predicted and true intensity (uses `lam_true`; synthetic-only
  diagnostic).

`tune` selects on `poisson_dev`. `Bench`-vs-`PoissonBench` handling: `evaluate` and
`run_grid` accept either (duck-typed: both expose `train`/`test`/`task`).

## Suite: `baselines_poisson`

10 models × 10 seeds on presets `poisson_k8` (500 objects) and `poisson_k64` (2000 objects,
k=64); tuned externals (8 trials); outputs the standard report bundle (results.csv, mean±std
md+tex tables for both metrics, Friedman/Nemenyi JSON, CD PDF, run_meta.json). Registered in
`SUITES` and the CLI.

## Testing

- **Semantics validation (gate)**: two groups, constant intensities 10 and 30 → basic-mode
  predictions ≈ 10/30; two groups with $\lambda(t)= 2 + 2t$ vs $4 - 2t$ → extra-mode
  predictions track the true rates within tolerance at several $t$. If this fails: STOP.
- generator: determinism, shapes, positivity of `lam_true`, mean count ≈ λΔ (statistical
  tolerance), event/tabular consistency (sum of freqs = sum of train counts).
- wrappers: fit/predict smoke on a tiny bench for every model.
- stats: poisson_dev of the oracle beats the marginal-mean predictor.

## Out of scope

- Engine (Go) changes; Poisson regime map; auto_roles for Poisson (basis-discovery caveat
  from the first round applies identically and is already documented);
- real-data Poisson experiments (falls under E3 stretch).

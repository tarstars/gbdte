# Article Experiments Design (ECML PKDD resubmission)

Date: 2026-07-02
Status: approved-by-default (user AFK; decisions below use the recommended options and are
open to revision on review)
Source of requirements: `~/prj/gbdte_article_2026/experiment_plan.md` (derived from the three
ECML PKDD Reject reviews) and `~/prj/gbdte_article_2026/review/gbdte_main_project.md`.

## Goal

Produce reproducible, paper-ready experimental evidence that answers the reviewers:

- E7 baseline matrix (incl. LightGBM `linear_tree=true`) on the existing benchmarks
- E6 statistical protocol: multi-seed runs, mean±std, Friedman + Nemenyi CD diagram
- E1 parametric benchmark family generalizing the current MSE/LogLoss generators
- E2 regime map: GBDTE gain as a function of role purity ρ and drift strength
- E4 data-computable role-separability / drift statistic, validated against ρ
- E5 automatic role assignment from the E4 statistic (oracle vs auto vs wrong vs none)
- E3 (stretch) real-data harness for TabReD / public credit data — loader interface only if
  downloads are infeasible from this machine

## Decisions taken while the user was away

1. **Code home: this repo** (`~/prj/extra_bridged_boosting`), branch
   `article-experiments-2026`. Rationale: it is the canonical, git-tracked implementation the
   paper cites; the article repo's `gbdte-main/` is a detached snapshot to be refreshed later.
2. **Scope order**: E7+E6 first (cheapest, converts draft prose into defensible tables), then
   E1+E2, then E4+E5, E3 last/stretch.
3. **Environment**: `uv venv` in-repo; deps: numpy, pandas, scipy, matplotlib, lightgbm,
   xgboost, catboost, scikit-learn, scikit-posthocs, pytest.
4. **GBDTE itself** is used through the existing Go engine via `extra_boost_py` (CGO build,
   go1.18). No engine changes are planned; experiments are a consumer layer.

## Architecture

New consumer-layer package `python/extra_boost_py/experiments/` plus one orchestration
script. The engine and existing pipeline are not modified.

```
python/extra_boost_py/experiments/
  __init__.py
  benchgen.py     # E1: parametric benchmark family (dataclass config -> DataFrame)
  baselines.py    # E7: unified model wrappers with a common fit/predict interface
  rolestat.py     # E4+E5: separability statistic + automatic role assignment
  stats.py        # E6: multi-seed runner, Friedman/Wilcoxon, CD-diagram plotting
  suites.py       # experiment definitions binding the above into named suites
scripts/run_article_experiments.py   # CLI: --suite <name> --seeds N --out reports/article_experiments
reports/article_experiments/         # md + tex + json tables, pdf (vector) figures
```

### benchgen.py (E1)

Data model: for each row, latent group `g = g(X_p)` from `k` groups encoded in binary
partition features; target

```
score(x, t) = φ(t)ᵀ β_g + ε          (regression: y = score;
                                       classification: y ~ Bernoulli(σ(score)))
```

`BenchConfig` knobs:

- `k_groups`: 8 … 1024 (presets `mse_small`≈8, `mse_big`≈128 reproduce the paper's story)
- `basis`: `"linear"` = (1, t), `"fourier"` = (1, t, sin ωt, cos ωt), ω configurable
- `rho` (role purity ∈ [0,1]): fraction of the signal that respects the partition /
  extrapolation split. Implementation: with weight (1−ρ) the group coefficients β_g are
  replaced by a component that depends on a *mixed* feature that also enters the leaf basis,
  so at ρ=0 partition features carry no clean group signal and extrapolation features leak
  into partitioning. ρ=1 reproduces the pure benchmarks.
- `drift`: scale of the time-dependent part of β (0 = no drift; the regime where linear
  leaves should NOT win)
- `noise`: ε std (regression) / label temperature (classification)
- `cut`: temporal train/test boundary (default 0.5 regression, 0.4 classification)
- `n_rows`, `seed`
- `task`: `"mse"` | `"logloss"`; for logloss the generator also returns the Bayes-optimal
  score for the oracle baseline.

Output: DataFrame with columns `f_*` (partition), `e_*` (extrapolation basis values),
`t`, `y`, plus metadata (true group, oracle score) for diagnostics.

### baselines.py (E7)

`class Model(Protocol): fit(train_df, spec) / predict(test_df) / name`. Wrappers:

- `GBDTE(roles="oracle" | "wrong" | "all_in_leaf")` — via `extra_boost_py.booster`;
  "wrong" swaps the blocks, "all_in_leaf" puts partition features into the leaf model
  (the tg_072 failure mode)
- `LightGBMModel(linear_tree: bool)`
- `XGBoostModel()`, `CatBoostModel()`
- `OracleModel()` (logloss only; uses generator-provided Bayes score)
- `GroupLinearModel()` (mse only; per-true-group regression on φ(t) — upper bound)

Uniform tuning policy: one shared small random-search budget (N=20 trials) over each
model's grid, tuned on a validation tail of the training window — never on the test window.
Metrics: RMSE (mse), ROC AUC + log-loss (logloss), plus time-sliced AUC reusing
`extra_boost_py.metrics` where applicable.

### rolestat.py (E4 + E5)

- `drift_score(feature, t)`: AUC of a stump/logistic predictor of `t > median(t)` from the
  single feature (model-free, cheap). High = feature drifts / carries time signal.
- `partition_score(feature, y, t)`: stability of the feature→y relation across time halves
  (sign-consistent correlation / per-half mutual information ratio).
- `separability_index(df)`: aggregate over features — designed so that on E1 data it is
  monotone in ρ (validated by test and by experiment).
- `auto_roles(df, threshold | top_k)` → (partition_features, extra_features) for E5.

### stats.py (E6)

- `run_grid(models, configs, seeds=10)` → tidy DataFrame of (model, config, seed, metric)
- mean±std tables → markdown + LaTeX (booktabs) + json
- Wilcoxon signed-rank for paired model comparisons; Friedman χ² + Nemenyi post-hoc via
  scikit-posthocs; CD diagram rendered with matplotlib to vector PDF (reviewer asked for
  vector graphics explicitly).

### suites.py + CLI

Named suites, each writing a self-contained report directory:

1. `baselines_mse`, `baselines_logloss` — E7×E6 on preset benchmarks
2. `regime_map` — E2: grid over (ρ, drift) × models → heatmap figure of relative test error
3. `rolestat_validation` — E4: separability index vs ρ curve; index computed per dataset of
   the regime map
4. `auto_roles` — E5: GBDTE with oracle/auto/wrong/none roles across the regime-map grid

## Error handling

- Engine build failure (CGO/go1.18): fail fast with a clear message; smoke-test the bridge
  first (`scripts/run_smoke_tests.py`) before long runs.
- Baseline package import errors: each wrapper degrades to "skipped" with a note in the
  report rather than aborting the suite.
- All randomness seeded; every report embeds the config JSON and git SHA.

## Testing

pytest under `python/tests/` (new):

- generator: shapes, determinism per seed, ρ=1 reproduces pure-role structure, oracle score
  achieves best log-loss on its own data
- rolestat: separability index monotone in ρ on generated grids (statistical tolerance)
- baselines: each wrapper fits/predicts on a 200-row dataset (skip if package missing)
- stats: CD-diagram/Friedman path runs on synthetic metric tables

TDD for the pure-Python units (benchgen, rolestat, stats); the wrappers get smoke tests.

## Out of scope (this iteration)

- Engine (Go) changes, Poisson mode
- Actual TabReD / credit-data downloads if network or size blocks them; the loader interface
  and one documented entry point are the stretch deliverable
- Writing the article text; the article repo consumes `reports/article_experiments/` later

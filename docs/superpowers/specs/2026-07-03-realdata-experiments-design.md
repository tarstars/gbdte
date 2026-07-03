# E3: Real-Data Experiments Design

Date: 2026-07-03
Status: approved-by-default (user AFK at the dataset-scope question; recommended option
taken, every decision open to revision)
Context: the last unanswered reviewer objection (R2/R3: no real-world data). Plan of
record: `~/prj/gbdte_article_2026/paper_progress.md`, "Agreed next move (2026-07-03)".

## Goal

Temporal-shift evaluation of GBDTE vs external boosters on public real datasets, with the
role-separability statistic (E4) computed *before training* to predict where GBDTE wins.
Mixed outcomes are acceptable and publishable under this framing.

## Decisions taken

1. **Datasets (first iteration)** — all verified downloadable with the user's Kaggle
   credentials (2026-07-03 probe; user already enrolled in both competitions):
   - `weather` (TabReD; Kaggle dataset `pcovkrd84mejm/tabred-weather`, 3.6 GB parquet,
     regression, 103 features, genuine timestamps) — subsampled;
   - `sberbank` (competition `sberbank-russian-housing-market`, regression, ~30k rows,
     `timestamp` column) — small, used whole;
   - `homecredit` (competition `home-credit-credit-risk-model-stability`, binary
     classification, `date_decision` timestamps) — the credit-scoring domain the paper's
     abstract names; light feature set (see 3).
2. **Light preprocessing, not TabReD reconstruction.** We need the temporal-shift
   *setting*, not TabReD's exact numbers: numeric features kept, categoricals
   ordinal-encoded, rows sorted by timestamp, normalized time `t in [0,1]`, temporal
   train/test split at the 0.75 quantile (train's last quarter is the tuning
   validation, as everywhere else). Documented per dataset in code.
3. **Home Credit feature set**: `train_base.csv` (case_id, date_decision, target) joined
   with the depth-0 static tables (`train_static_0_*`, `train_static_cb_0`). This is a
   deliberately simplified, fully documented feature set — not the competition's full
   multi-table reconstruction.
4. **Storage**: raw downloads and cached parquet extracts under `datasets/realdata/`
   (gitignored — multi-GB). Loaders download on first use via the `kaggle` CLI and cache
   a compact per-dataset parquet (subsampled where needed).
5. **Roles on real data**: no hand-made basis exists, so GBDTE runs with
   `auto_roles` (drift-score threshold): leaf block = (1, t) + detected drifting
   features; partition block = the rest. The real-data `Bench` provides `e_0 = 1` and
   `e_1 = t` columns so the existing `GBDTEModel("auto")` works unchanged. A
   `gbdte` (all features partition, (1,t) leaf) variant runs too.
6. **Subsampling = seeding**: each seed draws a fresh row subsample (default cap 60k
   rows per dataset, adjusted by a timing probe in the plan), so multi-seed statistics
   reflect sampling variability. 5 seeds for the first pass.
7. **Scope**: no Poisson, no engine changes. New code is a consumer layer as before.

## Architecture

```
python/extra_boost_py/experiments/
  realdata.py      # dataset registry, download/cache, per-dataset loader -> Bench
suites.py          # + suite_realdata (registered as "realdata")
scripts/run_article_experiments.py   # unchanged (suite auto-registered)
datasets/realdata/                   # gitignored cache
```

### realdata.py

- `REAL_DATASETS: dict[str, RealDatasetSpec]` — frozen dataclass per dataset:
  kaggle ref + type (dataset/competition), files to fetch, time column, target column,
  task ("mse" | "logloss"), loader function name.
- `load_real(name, seed, n_max=60000, cache_dir="datasets/realdata") -> Bench` —
  downloads on first call (kaggle CLI via subprocess), builds the cached full extract,
  then returns a seeded subsample as a standard `Bench`:
  - columns: ordinal-encoded/numeric features `x_*`, `t` (normalized time),
    `e_0`=1, `e_1`=t, `y`;
  - `partition_cols` = all feature columns, `extra_cols` = ["e_0","e_1"],
    `cut` = 0.75 time quantile, `task` per spec.
- Missing values: numeric NaN -> median (train-window median), categorical NaN -> own
  category. No target leakage: encoders/medians fitted on the train window only is NOT
  required for ordinal encoding / medians computed on features alone (no target used),
  so global fitting is acceptable and simpler; documented.

### Suite `realdata`

For each dataset x seed: compute `separability_index` and per-feature drift scores on
the train window (before any training), then run models
`gbdte` (designed = all-partition + (1,t) leaf), `gbdte_auto`, `gbdte_const`,
`lgbm`, `lgbm_linear`, `xgb`, `catboost` with the standard tuning budget; metrics
RMSE / AUC+logloss as per task. Report bundle as usual plus:
- `separability.csv` — per dataset x seed: separability index, top drifting features;
- `gain_vs_separability.pdf` — GBDTE-vs-LightGBM relative metric against the
  separability index, one point per dataset (the E4 validation on real data).

## Error handling

- Download failures (network, quota, not enrolled): loader raises with the exact kaggle
  command to run manually; suite marks the dataset "skipped" and continues.
- Timing guard: the plan includes a probe; if GBDTE on 60k rows x ~100 features exceeds
  ~5 min/fit, lower `n_max` (recorded in `run_meta.json`).

## Testing

- Unit tests with a synthetic frame exercising the loader transform (encoding, NaN
  handling, `t` normalization, split) without network.
- One integration test per available cache (skipped when `datasets/realdata/` absent) so
  CI-without-data still passes.
- Suite quick mode: tiny subsample (2k rows), 2 seeds, 2 models.

## Out of scope

- TabReD's full preprocessing pipelines and the other 3 large TabReD datasets;
- air-traffic data (optional later);
- writing results into the article (comes after numbers exist, per workflow).

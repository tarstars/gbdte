# ExtraBoost Classical Experiment (LOGLOSS loss)

Synthetic dataset generated via the legacy boolean feature recipe (see `classical_test_approach.md`).

## Configuration
````json
{
  "n_samples": 100000,
  "learning_rate": 0.2,
  "max_depth": 6,
  "n_stages": 250,
  "reg_lambda": 0.0001,
  "loss": "logloss",
  "threads": 20,
  "seed": 123,
  "alpha": 0.3,
  "train_time_cutoff": 0.4,
  "uplift_bins": 20,
  "auc_bins": 20,
  "uplift_min_count": 25,
  "auc_min_count": 50,
  "uplift_feature_indices": [
    0,
    1
  ],
  "auc_bin_edges": [
    0.0,
    0.1,
    0.2,
    0.30000000000000004,
    0.4,
    0.5,
    0.6000000000000001,
    0.7000000000000001,
    0.8,
    0.9,
    1.0
  ],
  "evaluation_scope": "dataset"
}
````

## Key Metrics
- overall ROC AUC: **0.657**
- earliest evaluated window (t≈0.050) AUC: 0.920
- latest evaluated window (t≈0.950) AUC: 0.633
- feature 1 (descending lift) uplift flips from 0.096 to -0.199
- train/test split: 39907 (39.91%) / 60093 (60.09%) samples

## Artifacts
- `classical_experiment_results.pdf` — ROC AUC and uplift curves
- `summary.json` — raw metrics dump
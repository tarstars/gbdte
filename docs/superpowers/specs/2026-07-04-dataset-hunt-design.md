# Systematic Hunt for a Public Dataset Where GBDTE Wins

Date: 2026-07-04. Status: approved-by-default (user AFK at the portfolio/method question;
recommended options taken, all revisable). Effort: max.

## Goal

Find at least one PUBLIC dataset where GBDTE (role-separated linear-leaf boosting) beats
traditional constant-leaf boosters AND survives strong baselines. Prior E3 result: 4
datasets, all lost, all at separability 0.0003-0.054 (below the >=0.08 synthetic-advantage
zone). Diagnostic is 4-for-4 at predicting negatives; we now hunt for a positive.

## Root-cause of the negatives (two, both actionable)

1. **The datasets lacked the mechanism.** GBDTE wins iff THREE conditions hold together:
   (a) stable, NONLINEAR group structure recoverable from features (high separability;
   nonlinear so a plain linear model can't substitute); (b) per-group APERIODIC drift with
   DIFFERENT slopes across groups (else a global time term suffices); (c) a forward
   temporal split creating an EXTRAPOLATION GAP (per-group target leaves its training
   range). Cross-sectional tabular benchmarks rarely have all three; PANEL/longitudinal
   data with entity-specific trends routinely does. This is textbook "trees can't
   extrapolate," localized per group.
2. **The diagnostic was blind.** `separability_index` scores only BINARY partition
   candidates (`np.unique(x).size <= 2`); high-cardinality/continuous group keys (country,
   sensor pattern, age-band) were skipped -> spurious ~0. Must be fixed regardless.

## Method: screen-first (compute-frugal, and itself a contribution)

Enrich the diagnostic, screen a candidate portfolio cheaply, run the expensive baseline
matrix ONLY where the screen is green. A green screen that then yields a win is the first
POSITIVE-direction validation of the diagnostic (so far only negatives + synthetic).

### Enriched diagnostics (`rolestat.py`)

- Fix `separability_index` to handle any-cardinality features: quantile-bin continuous /
  multi-valued x into <=10 bins, compute per-bin mean-y on early vs late time halves,
  score = Spearman rank-corr of per-bin means across halves (sign-consistent). Keep the
  binary path as a special case. Backward-compatible on synthetic benches (binary f_i).
- New `extrapolation_gain(bench, max_depth=4) -> float`: the mechanism, as a cheap screen.
  Fit a shallow decision tree on PARTITION features only (no time) to y on the train
  window; its leaves = discovered groups. Split the train window forward at its own 0.75
  time-quantile (inner-train / inner-forward). For each group, fit per-group constant vs
  per-group linear-in-t on inner-train; measure the metric improvement (RMSE drop, or
  logloss drop) of linear over constant on inner-forward. Return the aggregate relative
  improvement. High => linear leaves have forward headroom here; ~0 => no dataset trick
  will help. This is a mini-GBDTE-vs-const experiment used as a scalar screen.

### Candidate portfolio (public; panel data with entity trends + forward split)

Priority by mechanism strength and acquisition friction (no-auth first):

1. **UCI Gas Sensor Array Drift** (PRIMARY; https://archive.ics.uci.edu/static/public/224/,
   HTTP 200 no-auth). 13,910 rows, 128 continuous sensor features, 6 gas classes, 10
   time-ordered batches over 36 months. Classification. Group (gas class) is inferred
   NONLINEARLY from 128 features (trees essential, not a given key); sensors DRIFT over
   time per class (linear-leaf compensation); forward split by batch (train early, test
   late) is the standard hard protocol. Matches the project's own "attribute to class +
   model per-class drift compensation" framing (channel msg 188). Strongest clean fit.
2. **World Bank WDI country panel** (SECONDARY; api.worldbank.org, JSON no-auth). Pull a
   few strongly-trending indicators (child mortality, life expectancy, GDP per capita)
   for all countries x years. To keep the group INFERRED (not a country key): features =
   region, income group, and lagged/initial indicator levels; country identity withheld.
   Target = indicator value (or its level); forward split by year. Careful baseline set
   (below) required because per-country trends are near-linear.
3. **Human Mortality Database** (STRETCH; mortality.org, FREE REGISTRATION -> user action).
   age x country x year log-mortality; the Lee-Carter per-group linear-in-time
   extrapolation problem -- the strongest theoretical fit but gated behind a login. Note
   as a user-action blocker; do not block the branch on it.

### Baseline matrix (credibility guards)

Real-data model set extends the existing wrappers with two anti-strawman baselines so any
win is attributable to the leaf-linear MECHANISM, not to information or triviality:
- existing: gbdte, gbdte_auto, gbdte_const, lgbm, lgbm_linear, xgb, catboost;
- ADD `linear` (Ridge on partition features + t, with modest interactions) -- preempts
  "just use linear regression / a global time trend";
- ADD `detrend_lgbm` (fit a global linear-in-t trend, then LightGBM on residuals) --
  preempts "just detrend then boost".
GBDTE must beat gbdte_const (isolates the leaf model), lgbm/xgb/catboost (constant-leaf
boosters), AND linear/detrend_lgbm (isolates the need for nonlinear group discovery).

### Suite `hunt`

For each cached candidate: compute enriched separability + extrapolation_gain, write
`screen.csv` ranking. Then (same run or gated) run the full extended matrix on candidates
whose screen passes a threshold (extrapolation_gain relative improvement >= 5% AND
separability >= 0.05, tunable), 5 seeds, standard report bundle + the gain-vs-screen
figure. Respect GBDTE_REALDATA_ONLY.

## Testing

- separability_index: still monotone in rho on synthetic; NEW: recovers a stable
  high-cardinality categorical group (constructed frame) that the old binary-only code
  missed.
- extrapolation_gain: synthetic per-group linear-in-t signal with forward gap -> high;
  same signal with no per-group slope (only constant offsets) -> ~0; periodic-only signal
  -> ~0 (mechanism check).
- loaders: offline transform tests for gassensor (batch->t, class->y) and worldbank
  (JSON->frame, country withheld, lagged features), network-gated integration tests.

## Non-goals (this iteration)

HMD download automation, sales/demand de-seasonalization portfolio, semi-synthetic
drift injection (documented as the fallback if the hunt comes up empty).

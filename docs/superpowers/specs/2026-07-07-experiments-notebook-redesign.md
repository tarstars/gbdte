# Experiments Notebook Redesign

Date: 2026-07-07. Status: approved (design confirmed by user in brainstorm).
Replaces the mechanical `describe()`-and-scatter notebook at
`notebooks/experiments/all_dataset_experiments.ipynb`.

## Goal

A research-exploration notebook for the author and collaborators (the Telegram-channel
friends) that (a) lets someone unfamiliar with the datasets **understand each one deeply**
("everything we can get on them"), and (b) tells the coherent story of the claims with
polished, interactive, purpose-built visuals. Full redesign on three axes: narrative,
per-dataset insight, and visual quality.

## Audience & viewing

Author + technical collaborators, running locally or reading on **nbviewer** (interactive
Plotly renders there; GitHub's static viewer will not render interactive Plotly/widgets —
accepted, nbviewer is the share target). Widgets are optional local enhancers; the default
Plotly figures carry all content so the notebook is complete without a live kernel.

## Architecture: a tested helper module + a narrative notebook

Heavy lifting goes in a new, unit-tested module so the notebook stays a readable narrative.

### `python/extra_boost_py/experiments/eda.py`

- `RAW_INFO: dict[str, DatasetInfo]` — per-dataset metadata: plain-language description,
  source URL, task, real-world notes, and a `column_meanings: dict[str,str]` (human meaning
  of key columns; a generic fallback for the many-column datasets).
- `load_raw(name) -> pd.DataFrame` — the **raw** frame with original columns and real values
  (country names, gas classes, etc.), NOT the withheld/encoded modelling frame. For OWID
  this re-fetches the grapher CSV (entity, code, year, value) + continents and caches it
  under `datasets/realdata/<name>/raw_eda.parquet`; for the others it reads the existing
  `extract.parquet` cache (which already holds original-ish columns).
- Card builders returning Plotly figures / DataFrames (pure, testable):
  `overview_metrics(name)`, `column_table(df, name)` (name, dtype, %missing, #unique,
  range/example, meaning), `distribution_fig(df, col)`, `target_over_time_fig(name)`
  (real entities/segments, split line), `temporal_coverage_fig(df)`,
  `diagnostic_verdict(name)` (separability + extrap_gain + one-line why).
- A shared `dataviz`-derived Plotly template (`apply_style(fig)`) so every chart reads as
  one system (colors, fonts, gridlines, light/dark-safe). Implementation reads the
  `dataviz` skill first.

### The notebook `notebooks/experiments/all_dataset_experiments.ipynb`

Programmatically generated (nbformat) and executed (nbclient), as before. Structure:

**Act 0 — Intuition.** One toy chart: a single group's rising trend, the train/test split,
constant-leaf prediction flattening vs. linear-leaf extrapolating. The "aha" before formalism.

**Act 1 — The controllable world (synthetic).** MSE / LogLoss / Poisson generators each get a
dataset card (below) plus their ground-truth structure (groups, oracle, true intensity). The
regime map as an interactive ρ×drift explorer (widget locally; static heatmap otherwise) and a
live mini-demo of the two diagnostic numbers.

**Act 2 — The datasets (the heart).** A consistent **dataset card** for each of the 9 real
datasets (and the 3 synthetic in Act 1), in this order — winners and negatives interleaved by
the narrative but each fully profiled:

1. What it is — description, source link, real-world meaning, why it's in the study.
2. At a glance — records, time span, #entities/groups, task, target, train/test sizes.
3. A peek — first rows of the raw data (real names/values).
4. Every column — table: name, dtype, %missing, #unique, range/example, meaning.
5. Distributions — Plotly histograms (numeric) / top-category bars (categorical); an optional
   column-picker widget to flip through all columns.
6. The target — distribution, behaviour over time (key view), by segment/group.
7. Temporal structure — coverage over time, the forward split drawn on, drift/periodicity signature.
8. Diagnostic verdict — this dataset's separability + extrap_gain and the one-line why.

**Act 3 — The hunt & verdict.** The screen scatter (gain vs separability, hover) as hero
figure; the winners' trajectory-with-predictions charts (GBDTE tracks the trend, LightGBM
flattens); the family bootstrap bars with CIs (from `family_bootstrap.csv`); the
regularized-linear-tree stability point; honest caveats (detrend peer, GDP tie, Poisson scope).

**Appendix.** Full `describe()` per dataset (folded into the column tables; complete raw stats).

## Dependencies

Add to venv: `plotly`, `ipywidgets`, `kaleido` (static PNG export for hero figures reused in
the paper). `nbformat`/`nbclient` already present.

## Testing / verification

- Unit tests (`python/tests/test_eda.py`): `load_raw` returns expected original columns for a
  couple of datasets (OWID has country name; gassensor has gas + s1..s128); `column_table`
  emits one row per column with the required fields; `distribution_fig`/`target_over_time_fig`
  return Plotly `Figure` objects. Network-dependent raw loads are skipped when offline.
- Execute the notebook end-to-end (nbclient): 0 error outputs; every figure present; spot-check
  the hero numbers (child mortality GBDTE < LightGBM). Verify visually via a rendered export.

## Out of scope

- Paper-grade static rendering (nbviewer is the target); re-running expensive tuning (load the
  committed `family_bootstrap.csv`); transaction-level raw for LatAm beyond the aggregated card.

## Reproducibility

Notebook committed executed; raw OWID re-fetch cached to `raw_eda.parquet`; all model numbers
trace to committed reports. Lives in the code repo (experiments artifact).

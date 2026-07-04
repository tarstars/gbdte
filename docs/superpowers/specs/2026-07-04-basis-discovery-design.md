# Discovered Extrapolation Basis for Real Data (E3 extension)

Date: 2026-07-04. Status: approved (user chose all three options in brainstorm:
periodogram->Fourier+trend, basis given to all models, weather-first-then-all-four;
then delegated remaining judgment).

## Motivation (user's observation)

Real datasets do not ship extrapolating features; expecting them is naive. Dataset
features act as interpolating/partition features, and the extrapolation basis must be
CONSTRUCTED from the data: preliminary spectral analysis (Fourier/periodogram; Prony and
SVD as alternatives; Prophet rejected as a heavy dependency that is trend+Fourier
internally). Current E3 used the naive (1, t) leaf basis — GBDTE could not express the
periodic structure real targets actually have.

## Design

### `experiments/basisdisc.py`

- `DiscoveredBasis` dataclass: `freqs: tuple[float, ...]` (cycles per unit of normalized
  t), `r2_train: float` (basis R^2 on the aggregate train curve), `periods_natural:
  tuple[str, ...]` (human-readable, using the bench's time span if known).
- `discover_basis(bench, max_freqs=3, n_bins=200, min_power_ratio=4.0) -> DiscoveredBasis`
  - TRAIN WINDOW ONLY (leakage-clean).
  - aggregate curve: mean y per t-bin (bins with data only);
  - detrend: subtract OLS line;
  - Lomb-Scargle periodogram (scipy.signal.lombscargle) on frequency grid
    [2 / span_train, n_bins / (4 * span_train)] (well below bin Nyquist), 2000 grid pts;
  - greedy peak pick: highest power first, >= min_power_ratio x median power, minimum
    separation 2 grid steps; at most max_freqs;
  - r2_train: OLS of the aggregate curve on [1, t, sin/cos(2 pi f t) for f].
- `apply_basis(bench, basis) -> Bench`: new Bench (deep-copied df) with
  `e_{2i+2}=sin(2 pi f_i t)`, `e_{2i+3}=cos(2 pi f_i t)` appended to `extra_cols` AND
  the same columns appended to `partition_cols` (fairness: external boosters see them
  as ordinary features).

### Suite `realdata_basis`

Same models/protocol as `realdata` (7 models, tuned, 5 seeds, separability recorded),
but each seed's bench is wrapped by discover_basis/apply_basis; run_meta.json gains
per-dataset per-seed discovered frequencies and r2_train. Respects GBDTE_REALDATA_ONLY.
Output: `reports/article_experiments/realdata_basis/`.

### Execution order

1. weather only (GBDTE_REALDATA_ONLY=weather) — machinery validation; inspect
   discovered periods (expect diurnal and/or annual) and results;
2. full four-dataset run;
3. extend gain-vs-separability analysis with a second predictor axis: basis r2_train.

## Testing

- synthetic: y = 0.5 t + sin(2 pi 8 t) + noise -> discover_basis finds f close to 8, r2
  high; pure-noise signal -> no frequencies pass the floor;
- leakage: signal with frequency 8 for t < cut and 20 for t >= cut -> discovered
  frequency is 8 only;
- apply_basis: column bookkeeping (extra_cols and partition_cols extended; df columns
  present and finite; original bench unmodified).

## Out of scope

Prony analysis and SVD/functional-PCA basis (documented as follow-ups), Prophet,
synthetic-suite integration (the synthetic benches already have known bases).

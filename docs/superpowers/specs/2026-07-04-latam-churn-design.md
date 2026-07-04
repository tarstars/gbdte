# LatAm Fintech Dataset for E3 (addendum)

Date: 2026-07-04. Status: approved (user said "yes" to the approach described in
`gbdte_article_2026/review/datasets_from_chat.md`); details below decided by me.

## Source

Mendeley `mhb4zn3258/1` ("Customer Behavior in Latin American Fintech"): public direct
downloads, no auth. `customer_data.csv` (48.7k clients, 52 fields),
`transactions_data.csv` (customer_id, date, amount, type; 12 months of 2023).

## Task formulation: monthly activity forecasting (MSE)

Chosen over churn classification: churn labels have no per-row timestamp (and the card's
`churn_probability` is of unknown provenance — the chat flags it), while activity
forecasting is exactly GBDTE-shaped: static client card = partition features, months =
time, per-client monthly transaction count = drifting target (churn appears as decaying
activity). Construction:

- grid rows = client x month (2023-01..2023-12), zero months materialized (crucial:
  churn = trailing zeros);
- `y = log1p(monthly transaction count)`; `time_col` = month start;
- features: **whitelisted static-only card fields** — age, gender, location,
  income_bracket, occupation, education_level, marital_status, household_size,
  acquisition_channel, customer_segment, savings_account, credit_card, personal_loan,
  investment_account, insurance_product, active_products. Everything else in the card
  (tx_count, avg_tx_value, first/last_tx, app usage, churn_probability, ...) is a
  whole-2023 aggregate = future leakage under a temporal split → excluded;
- standard pipeline after that: `frame_to_bench` (0.75 time-quantile split), registry
  entry `latam` with a custom reader; separability computed pre-training; the existing
  `realdata` suite picks it up automatically via the cache.

## Non-goals now

Poisson-mode run on the same aggregation (natural follow-up), churn classification,
amount forecasting.

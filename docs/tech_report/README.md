# GBDTE Technical Report (Outline)

## Abstract
We present GBDTE, a Go+Python implementation of ExtraBoost-style gradient boosting for extrapolation-focused research.
The system pairs a fast Go learner with a thin Python bridge for datasets, metrics, and experiment orchestration.
We report classical MSE/logloss benchmarks and time-sliced evaluation workflows for reproducible comparisons.
The repository includes scripts and reports that make results auditable and easy to extend.
This document will expand into a 4–6 page technical report with formal experiments and ablations.
We highlight design choices that balance research iteration speed with reproducibility.

## Key contributions
- Hybrid Go/Python architecture for fast training with reproducible experiment scripts.
- Classical benchmark suite with stored reports for MSE/logloss comparisons.
- Time-sliced evaluation utilities for extrapolation and distribution shift analysis.

## Reproduction
Follow the steps in the "Results and Reproducing Figure 1" section of `README.md`.

## Data and reports
- Datasets live under `datasets/`.
- Generated reports are written to `reports/classical_experiment/`.

## How to cite
See `CITATION.cff` in the repository root.

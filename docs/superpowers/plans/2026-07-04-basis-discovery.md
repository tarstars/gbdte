# Basis Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Periodogram-discovered Fourier+trend leaf basis for real datasets, given to all models, validated on weather then run on all four datasets.

**Architecture:** New `experiments/basisdisc.py` (pure, offline-testable); `suite_realdata_basis` in `suites.py` wrapping each seed's bench with the discovered basis; reports as usual.

**Tech Stack:** scipy.signal.lombscargle, existing pipeline.

## Global Constraints

- Discovery uses the train window only (leakage test enforces it).
- Both `extra_cols` and `partition_cols` receive the new columns (fairness decision).
- Respect `GBDTE_REALDATA_ONLY`; record freqs + r2_train in run_meta.json.

---

### Task 1: basisdisc.py (TDD)

**Files:** Create `python/extra_boost_py/experiments/basisdisc.py`; Test `python/tests/test_basisdisc.py`.

**Interfaces:** `DiscoveredBasis(freqs, r2_train)`, `discover_basis(bench, max_freqs=3, n_bins=200, min_power_ratio=4.0)`, `apply_basis(bench, basis) -> Bench` per spec.

Steps: write the three test groups from the spec (recovery of f=8, noise rejection,
leakage isolation, apply_basis bookkeeping); verify FAIL; implement; verify PASS; commit
"Add periodogram basis discovery (train-window only)".

### Task 2: suite + runs

**Files:** Modify `python/extra_boost_py/experiments/suites.py` (suite_realdata_basis,
register "realdata_basis"), `python/tests/test_suites.py` (quick test, skip w/o cache).

Steps: failing quick test; implement (clone of suite_realdata with basis wrap + meta);
full pytest green; commit "Add realdata_basis suite". Run weather-only 5 seeds
(background), inspect discovered periods and summary; then full 4-dataset run; merge
gain-vs-separability-vs-r2 analysis; commit reports; update paper_progress; finish
branch (merge to main + push, per standing precedent).

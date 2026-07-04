from pathlib import Path

import pytest

from extra_boost_py.experiments.suites import run_suite
from extra_boost_py.experiments.suites import _cached_real_names, run_suite


def test_quick_realdata(tmp_path: Path):
    if not _cached_real_names():
        pytest.skip("no real-data caches present")
    out = run_suite("realdata", tmp_path, seeds=2, quick=True)
    assert (out / "results.csv").exists()
    assert (out / "separability.csv").exists()


def test_quick_regime_map(tmp_path: Path):
    out = run_suite("regime_map", tmp_path, seeds=2, quick=True)
    assert (out / "results.csv").exists()
    assert (out / "run_meta.json").exists()


def test_quick_baselines_poisson(tmp_path: Path):
    out = run_suite("baselines_poisson", tmp_path, seeds=2, quick=True)
    assert (out / "results.csv").exists()
    assert (out / "summary_poisson_dev.md").exists()


def test_quick_rolestat(tmp_path: Path):
    out = run_suite("rolestat_validation", tmp_path, seeds=2, quick=True)
    assert (out / "separability_vs_rho.pdf").exists()


def test_quick_realdata_basis(tmp_path: Path):
    if not _cached_real_names():
        pytest.skip("no real-data caches present")
    out = run_suite("realdata_basis", tmp_path, seeds=1, quick=True)
    assert (out / "results.csv").exists()
    import json
    meta = json.loads((out / "run_meta.json").read_text())
    assert "discovered" in meta

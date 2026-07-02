from pathlib import Path

from extra_boost_py.experiments.suites import run_suite


def test_quick_regime_map(tmp_path: Path):
    out = run_suite("regime_map", tmp_path, seeds=2, quick=True)
    assert (out / "results.csv").exists()
    assert (out / "run_meta.json").exists()


def test_quick_rolestat(tmp_path: Path):
    out = run_suite("rolestat_validation", tmp_path, seeds=2, quick=True)
    assert (out / "separability_vs_rho.pdf").exists()

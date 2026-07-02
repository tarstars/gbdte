from pathlib import Path

import numpy as np
import pandas as pd

from extra_boost_py.experiments.stats import (
    cd_diagram, friedman_nemenyi, summary_table)


def _fake_results():
    rng = np.random.default_rng(0)
    rows = []
    for bench in ["b1", "b2", "b3", "b4", "b5", "b6"]:
        for model, mu in [("good", 0.1), ("mid", 0.2), ("bad", 0.4)]:
            for seed in range(5):
                rows.append(dict(bench=bench, model=model, seed=seed,
                                 metric="rmse", value=mu + 0.01 * rng.standard_normal()))
    return pd.DataFrame(rows)


def test_summary_table():
    tab = summary_table(_fake_results(), "rmse")
    assert set(tab.columns) == {"good", "mid", "bad"}
    assert "±" in tab.iloc[0, 0]


def test_friedman_detects_difference():
    res = friedman_nemenyi(_fake_results(), "rmse", higher_is_better=False)
    assert res["p_value"] < 0.01
    assert res["avg_ranks"]["good"] < res["avg_ranks"]["bad"]


def test_cd_diagram_writes_pdf(tmp_path: Path):
    out = tmp_path / "cd.pdf"
    cd_diagram(_fake_results(), "rmse", False, out)
    assert out.exists() and out.stat().st_size > 0

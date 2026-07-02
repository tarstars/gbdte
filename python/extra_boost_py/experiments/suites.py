"""Named experiment suites producing paper-ready reports."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import make_models
from .benchgen import Bench, PRESETS, generate
from .classical_bench import classical_bench
from .rolestat import separability_index
from .stats import cd_diagram, friedman_nemenyi, run_grid, summary_table

RHO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
DRIFT_GRID = [0.0, 0.5, 1.0, 2.0]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_report(out: Path, results: pd.DataFrame, meta: dict,
                  higher_is_better: Dict[str, bool]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    for metric, hib in higher_is_better.items():
        if not (results["metric"] == metric).any():
            continue
        tab = summary_table(results, metric)
        tab.to_markdown(out / f"summary_{metric}.md")
        tab.to_latex(out / f"table_{metric}.tex")
        n_models = results[results["metric"] == metric]["model"].nunique()
        if n_models >= 3:
            st = friedman_nemenyi(results, metric, hib)
            (out / f"stats_{metric}.json").write_text(json.dumps(st, indent=2))
            cd_diagram(results, metric, hib, out / f"cd_{metric}.pdf")


def _seeds(n: int) -> List[int]:
    return list(range(n))


def suite_baselines_mse(out: Path, seeds: int, quick: bool) -> None:
    presets = {k: PRESETS[k] for k in (["mse_k8"] if quick else ["mse_k8", "mse_k128"])}
    models = make_models("mse")

    def factory(name: str, seed: int) -> Bench:
        cfg = replace(presets[name], seed=seed,
                      n_rows=400 if quick else presets[name].n_rows)
        return generate(cfg)

    res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                   bench_names=list(presets), tune_trials=2 if quick else 8)
    _write_report(out, res, {"suite": "baselines_mse", "git": _git_sha(),
                             "presets": {k: asdict(v) for k, v in presets.items()},
                             "seeds": seeds}, {"rmse": False})


def suite_baselines_logloss(out: Path, seeds: int, quick: bool) -> None:
    models = make_models("logloss")

    def factory(name: str, seed: int) -> Bench:
        return classical_bench(n_rows=400 if quick else 10000, seed=seed)

    res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                   bench_names=["classical"], tune_trials=2 if quick else 8)
    _write_report(out, res, {"suite": "baselines_logloss", "git": _git_sha(),
                             "seeds": seeds}, {"auc": True, "logloss": False})


def _regime_results(seeds: int, quick: bool, model_names: List[str]) -> pd.DataFrame:
    base = PRESETS["regime_base"]
    all_models = make_models("mse")
    models = {k: all_models[k] for k in model_names}
    names = [f"rho{rho}_drift{dr}" for rho in RHO_GRID for dr in DRIFT_GRID]
    if quick:
        names = names[:4]

    def factory(name: str, seed: int) -> Bench:
        rho = float(name.split("_")[0][3:])
        dr = float(name.split("_")[1][5:])
        cfg = replace(base, rho=rho, drift=dr, seed=seed,
                      n_rows=400 if quick else base.n_rows)
        return generate(cfg)

    return run_grid(models, None, _seeds(seeds), bench_factory=factory,
                    bench_names=names, tune_trials=0)


def suite_regime_map(out: Path, seeds: int, quick: bool) -> None:
    res = _regime_results(seeds, quick, ["gbdte", "lgbm"])
    _write_report(out, res, {"suite": "regime_map", "git": _git_sha(), "seeds": seeds,
                             "rho_grid": RHO_GRID, "drift_grid": DRIFT_GRID},
                  {"rmse": False})
    sub = res[res["metric"] == "rmse"].groupby(["bench", "model"])["value"].mean().unstack()
    if {"gbdte", "lgbm"} <= set(sub.columns):
        ratio = np.full((len(RHO_GRID), len(DRIFT_GRID)), np.nan)
        for i, rho in enumerate(RHO_GRID):
            for j, dr in enumerate(DRIFT_GRID):
                name = f"rho{rho}_drift{dr}"
                if name in sub.index:
                    ratio[i, j] = sub.loc[name, "gbdte"] / sub.loc[name, "lgbm"]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(np.log2(ratio), cmap="RdBu", vmin=-2, vmax=2, origin="lower",
                       aspect="auto")
        ax.set_xticks(range(len(DRIFT_GRID)), [str(d) for d in DRIFT_GRID])
        ax.set_yticks(range(len(RHO_GRID)), [str(r) for r in RHO_GRID])
        ax.set_xlabel("drift strength")
        ax.set_ylabel(r"role purity $\rho$")
        fig.colorbar(im, ax=ax, label=r"$\log_2$ RMSE(GBDTE)/RMSE(LightGBM)")
        fig.tight_layout()
        fig.savefig(out / "regime_map.pdf")
        plt.close(fig)


def suite_rolestat_validation(out: Path, seeds: int, quick: bool) -> None:
    base = PRESETS["regime_base"]
    rows = []
    for rho in RHO_GRID:
        for seed in _seeds(seeds):
            cfg = replace(base, rho=rho, seed=seed, n_rows=400 if quick else base.n_rows)
            rows.append(dict(rho=rho, seed=seed,
                             separability=separability_index(generate(cfg))))
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps(
        {"suite": "rolestat_validation", "git": _git_sha(), "seeds": seeds}, indent=2))
    agg = df.groupby("rho")["separability"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker="o", capsize=3)
    ax.set_xlabel(r"role purity $\rho$")
    ax.set_ylabel("separability index")
    fig.tight_layout()
    fig.savefig(out / "separability_vs_rho.pdf")
    plt.close(fig)


def suite_auto_roles(out: Path, seeds: int, quick: bool) -> None:
    res = _regime_results(seeds, quick,
                          ["gbdte", "gbdte_auto", "gbdte_wrong", "gbdte_const"])
    _write_report(out, res, {"suite": "auto_roles", "git": _git_sha(), "seeds": seeds},
                  {"rmse": False})


SUITES = {
    "baselines_mse": suite_baselines_mse,
    "baselines_logloss": suite_baselines_logloss,
    "regime_map": suite_regime_map,
    "rolestat_validation": suite_rolestat_validation,
    "auto_roles": suite_auto_roles,
}


def run_suite(name: str, out_root: Path, seeds: int = 10, quick: bool = False) -> Path:
    out = Path(out_root) / name
    SUITES[name](out, seeds, quick)
    return out


__all__ = ["SUITES", "run_suite", "RHO_GRID", "DRIFT_GRID"]

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
from .poisson_baselines import make_poisson_models
from .poisson_bench import POISSON_PRESETS, generate_poisson
from .basisdisc import apply_basis, discover_basis
from .realdata import REAL_DATASETS, load_real
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


def suite_baselines_poisson(out: Path, seeds: int, quick: bool) -> None:
    presets = {k: POISSON_PRESETS[k]
               for k in (["poisson_k8"] if quick else ["poisson_k8", "poisson_k64"])}
    models = make_poisson_models()

    def factory(name: str, seed: int):
        cfg = replace(presets[name], seed=seed,
                      n_objects=60 if quick else presets[name].n_objects,
                      n_bins=8 if quick else presets[name].n_bins)
        return generate_poisson(cfg)

    res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                   bench_names=list(presets), tune_trials=2 if quick else 8)
    _write_report(out, res, {"suite": "baselines_poisson", "git": _git_sha(),
                             "presets": {k: asdict(v) for k, v in presets.items()},
                             "seeds": seeds},
                  {"poisson_dev": False, "rate_rmse": False})


def _cached_real_names() -> List[str]:
    return [n for n in REAL_DATASETS
            if (Path("datasets/realdata") / n / "extract.parquet").exists()]


def suite_realdata(out: Path, seeds: int, quick: bool) -> None:
    cached = _cached_real_names()
    if not cached:
        raise RuntimeError("no real-data caches under datasets/realdata/; "
                           "run load_real() for at least one dataset first")
    import os
    only = os.environ.get("GBDTE_REALDATA_ONLY")
    if only:
        cached = [n for n in cached if n in only.split(",")]
    names = cached[:1] if quick else cached
    n_max = 2000 if quick else 20000   # timing probe 2026-07-03: engine ~45s/6k rows
    wanted = ["gbdte", "gbdte_auto", "gbdte_const", "lgbm", "lgbm_linear",
              "xgb", "catboost"]
    sep_rows, all_results = [], []
    for name in names:
        task = REAL_DATASETS[name].task
        all_models = make_models(task, include_oracle=False)
        models = {k: v for k, v in all_models.items() if k in wanted}

        def factory(bname: str, seed: int) -> Bench:
            bench = load_real(bname, seed=seed, n_max=n_max)
            sep_rows.append(dict(bench=bname, seed=seed,
                                 separability=separability_index(bench)))
            return bench

        res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                       bench_names=[name], tune_trials=2 if quick else 8)
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    hib = {"rmse": False, "auc": True, "logloss": False}
    _write_report(out, results, {"suite": "realdata", "git": _git_sha(),
                                 "seeds": seeds, "n_max": n_max,
                                 "datasets": names}, hib)
    sep = pd.DataFrame(sep_rows).drop_duplicates(["bench", "seed"])
    sep.to_csv(out / "separability.csv", index=False)

    rows = []
    for name in names:
        metric = "rmse" if REAL_DATASETS[name].task == "mse" else "logloss"
        sub = results[(results["bench"] == name) & (results["metric"] == metric)]
        perf = sub.groupby("model")["value"].mean()
        if {"gbdte_auto", "lgbm"} <= set(perf.index):
            rows.append(dict(bench=name, metric=metric,
                             rel=perf["gbdte_auto"] / perf["lgbm"],
                             separability=sep[sep["bench"] == name]["separability"].mean()))
    gain = pd.DataFrame(rows)
    gain.to_csv(out / "gain_vs_separability.csv", index=False)
    if len(gain) >= 2:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.scatter(gain["separability"], gain["rel"])
        for _, r in gain.iterrows():
            ax.annotate(f' {r["bench"]}', (r["separability"], r["rel"]), fontsize=8)
        ax.axhline(1.0, color="0.6", lw=1, ls="--")
        ax.set_xlabel("separability index (pre-training)")
        ax.set_ylabel("GBDTE / LightGBM (lower = GBDTE wins)")
        fig.tight_layout()
        fig.savefig(out / "gain_vs_separability.pdf")
        plt.close(fig)


def suite_realdata_basis(out: Path, seeds: int, quick: bool) -> None:
    """realdata with a per-seed DISCOVERED Fourier+trend basis given to all models."""
    import os
    cached = _cached_real_names()
    only = os.environ.get("GBDTE_REALDATA_ONLY")
    if only:
        cached = [n for n in cached if n in only.split(",")]
    if not cached:
        raise RuntimeError("no real-data caches under datasets/realdata/")
    names = cached[:1] if quick else cached
    n_max = 2000 if quick else 20000
    wanted = ["gbdte", "gbdte_auto", "gbdte_const", "lgbm", "lgbm_linear",
              "xgb", "catboost"]
    sep_rows, all_results, discovered = [], [], {}
    for name in names:
        task = REAL_DATASETS[name].task
        all_models = make_models(task, include_oracle=False)
        models = {k: v for k, v in all_models.items() if k in wanted}

        def factory(bname: str, seed: int) -> Bench:
            bench = load_real(bname, seed=seed, n_max=n_max)
            basis = discover_basis(bench)
            discovered[f"{bname}/seed{seed}"] = {
                "freqs": list(basis.freqs), "r2_train": basis.r2_train}
            bench = apply_basis(bench, basis)
            sep_rows.append(dict(bench=bname, seed=seed,
                                 separability=separability_index(bench),
                                 basis_r2=basis.r2_train,
                                 n_freqs=len(basis.freqs)))
            return bench

        res = run_grid(models, None, _seeds(seeds), bench_factory=factory,
                       bench_names=[name], tune_trials=2 if quick else 8)
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    hib = {"rmse": False, "auc": True, "logloss": False}
    _write_report(out, results, {"suite": "realdata_basis", "git": _git_sha(),
                                 "seeds": seeds, "n_max": n_max, "datasets": names,
                                 "discovered": discovered}, hib)
    pd.DataFrame(sep_rows).drop_duplicates(["bench", "seed"]).to_csv(
        out / "separability.csv", index=False)


SUITES = {
    "baselines_mse": suite_baselines_mse,
    "baselines_logloss": suite_baselines_logloss,
    "baselines_poisson": suite_baselines_poisson,
    "realdata": suite_realdata,
    "realdata_basis": suite_realdata_basis,
    "regime_map": suite_regime_map,
    "rolestat_validation": suite_rolestat_validation,
    "auto_roles": suite_auto_roles,
}


def run_suite(name: str, out_root: Path, seeds: int = 10, quick: bool = False) -> Path:
    out = Path(out_root) / name
    SUITES[name](out, seeds, quick)
    return out


__all__ = ["SUITES", "run_suite", "RHO_GRID", "DRIFT_GRID"]

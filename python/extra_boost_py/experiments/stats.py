"""E6: multi-seed evaluation, uniform tuning, Friedman/Nemenyi and CD diagram."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import log_loss, roc_auc_score

from .baselines import Model
from .benchgen import Bench


def evaluate(model: Model, bench: Bench) -> Dict[str, float]:
    test = bench.test
    pred = model.predict(test)
    y = test["y"].to_numpy()
    if bench.task == "mse":
        return {"rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}
    prob = 1.0 / (1.0 + np.exp(-pred))
    return {"auc": float(roc_auc_score(y, prob)),
            "logloss": float(log_loss(y, np.clip(prob, 1e-12, 1 - 1e-12)))}


def _val_split(bench: Bench) -> Tuple[Bench, Bench]:
    """Last-by-t 25% of the train window becomes validation."""
    tr = bench.train
    q = float(tr["t"].quantile(0.75))
    inner = Bench(df=tr[tr["t"] < q], partition_cols=bench.partition_cols,
                  extra_cols=bench.extra_cols, task=bench.task, cut=q)
    val = Bench(df=tr, partition_cols=bench.partition_cols,
                extra_cols=bench.extra_cols, task=bench.task, cut=q)
    return inner, val


def tune(model: Model, bench: Bench, n_trials: int = 8, seed: int = 0) -> dict:
    grid = model.param_grid()
    if not grid or n_trials <= 0:
        return {}
    rng = np.random.default_rng(seed)
    combos = list(itertools.product(*grid.values()))
    rng.shuffle(combos)
    key = "rmse" if bench.task == "mse" else "logloss"
    inner, val = _val_split(bench)
    best, best_score = {}, np.inf
    for combo in combos[:n_trials]:
        params = dict(zip(grid.keys(), combo))
        model.fit(inner, params)
        score = evaluate(model, val)[key]
        if score < best_score:
            best, best_score = params, score
    return best


def run_grid(models: Dict[str, Model], benches: Optional[Dict[str, Bench]],
             seeds: List[int],
             bench_factory: Optional[Callable[[str, int], Bench]] = None,
             bench_names: Optional[List[str]] = None,
             tune_trials: int = 8) -> pd.DataFrame:
    """Tidy results over benches x seeds x models. Benches are either fixed per name
    or regenerated per seed via bench_factory(name, seed)."""
    names = bench_names if bench_names is not None else list(benches.keys())
    rows = []
    tuned: Dict[tuple, dict] = {}
    for bname in names:
        for seed in seeds:
            bench = bench_factory(bname, seed) if bench_factory else benches[bname]
            for mname, model in models.items():
                if not model.available():
                    rows.append(dict(bench=bname, model=mname, seed=seed,
                                     metric="skipped", value=np.nan))
                    continue
                if (bname, mname) not in tuned:
                    tuned[(bname, mname)] = tune(model, bench, tune_trials, seed)
                model.fit(bench, tuned[(bname, mname)])
                for metric, value in evaluate(model, bench).items():
                    rows.append(dict(bench=bname, model=mname, seed=seed,
                                     metric=metric, value=value))
    return pd.DataFrame(rows)


def summary_table(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = results[results["metric"] == metric]
    agg = sub.groupby(["bench", "model"])["value"].agg(["mean", "std"]).reset_index()
    agg["cell"] = agg.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    return agg.pivot(index="bench", columns="model", values="cell")


def _rank_table(results: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    sub = results[results["metric"] == metric]
    perf = sub.groupby(["bench", "seed", "model"])["value"].mean().unstack("model")
    return perf.rank(axis=1, ascending=not higher_is_better)


def friedman_nemenyi(results: pd.DataFrame, metric: str, higher_is_better: bool) -> dict:
    ranks = _rank_table(results, metric, higher_is_better)
    cols = list(ranks.columns)
    stat, p = sps.friedmanchisquare(*[ranks[c].to_numpy() for c in cols])
    out = {"statistic": float(stat), "p_value": float(p),
           "avg_ranks": ranks.mean().to_dict(), "n_blocks": len(ranks)}
    try:
        import scikit_posthocs as sp
        sub = results[results["metric"] == metric]
        perf = sub.groupby(["bench", "seed", "model"])["value"].mean().reset_index()
        wide = perf.pivot(index=["bench", "seed"], columns="model", values="value")
        wide = wide.reset_index(drop=True)  # scikit-posthocs needs plain block rows
        out["nemenyi_p"] = sp.posthoc_nemenyi_friedman(wide).to_dict()
    except ImportError:
        out["nemenyi_p"] = None
    return out


def cd_diagram(results: pd.DataFrame, metric: str, higher_is_better: bool,
               out_pdf: Path) -> None:
    ranks = _rank_table(results, metric, higher_is_better)
    avg = ranks.mean().sort_values()
    k, n = len(avg), len(ranks)
    q_alpha = sps.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2.0)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    fig, ax = plt.subplots(figsize=(7, 0.6 * k + 1.2))
    y = np.arange(k)[::-1]
    ax.hlines(y, xmin=1, xmax=avg.to_numpy(), color="0.7", lw=1)
    ax.plot(avg.to_numpy(), y, "o", color="C0")
    for yi, (name, r) in zip(y, avg.items()):
        ax.annotate(f"  {name} ({r:.2f})", (r, yi), va="center", fontsize=9)
    ax.plot([1, 1 + cd], [k - 0.4] * 2, lw=3, color="C3")
    ax.annotate(f"CD = {cd:.2f}", (1, k - 0.15), color="C3", fontsize=9)
    ax.set_xlabel(f"average rank ({metric})")
    ax.set_yticks([])
    ax.set_xlim(0.8, k + 0.8)
    ax.set_ylim(-0.8, k)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


__all__ = ["evaluate", "tune", "run_grid", "summary_table",
           "friedman_nemenyi", "cd_diagram"]

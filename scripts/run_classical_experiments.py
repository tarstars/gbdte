from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from extra_boost_py.pipeline import ExperimentConfig, PipelineReport, run_classical_pipeline


def _to_series(data: Iterable[Tuple[Tuple[float, float], float]]) -> List[Tuple[float, float]]:
    series: List[Tuple[float, float]] = []
    for (left, right), value in data:
        center = 0.5 * (left + right)
        series.append((center, float(value) if not math.isnan(value) else math.nan))
    return series


def _first_valid(series: List[Tuple[float, float]]) -> Tuple[float, float] | None:
    for point in series:
        if not math.isnan(point[1]):
            return point
    return None


def _last_valid(series: List[Tuple[float, float]]) -> Tuple[float, float] | None:
    for point in reversed(series):
        if not math.isnan(point[1]):
            return point
    return None


def _write_report(base_path: Path, report: PipelineReport, flavor: str) -> None:
    base_path.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "config": asdict(report.config),
        "dataset_summary": report.dataset_summary,
        "results": {
            "overall_auc": report.results.overall_auc,
            "time_slice_auc": report.results.time_slice_auc,
            "uplift_feature_0": report.results.uplift_feature_0,
            "uplift_feature_1": report.results.uplift_feature_10,
        },
    }
    (base_path / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    auc_series = _to_series(report.results.time_slice_auc)
    uplift_a_series = _to_series(report.results.uplift_feature_0)
    uplift_b_series = _to_series(report.results.uplift_feature_10)

    fig, (ax_auc, ax_uplift) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    auc_centers = np.array([pt[0] for pt in auc_series], dtype=float)
    auc_values = np.array([pt[1] for pt in auc_series], dtype=float)
    ax_auc.plot(auc_centers, auc_values, marker="o")
    ax_auc.set_ylabel("ROC AUC")
    ax_auc.set_ylim(0.3, 1.0)
    ax_auc.set_xlim(0.0, 1.0)
    ax_auc.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax_auc.grid(True, alpha=0.3)
    ax_auc.set_title(f"ExtraBoost Classical Experiment — {flavor.upper()} loss")

    uplift_centers = np.array([pt[0] for pt in uplift_a_series], dtype=float)
    uplift_a_vals = np.array([pt[1] for pt in uplift_a_series], dtype=float)
    uplift_b_vals = np.array([pt[1] for pt in uplift_b_series], dtype=float)
    ax_uplift.plot(uplift_centers, uplift_a_vals, marker="o", label="Feature 0 (ascending)")
    ax_uplift.plot(uplift_centers, uplift_b_vals, marker="s", label="Feature 1 (descending)")
    ax_uplift.set_xlabel("time")
    ax_uplift.set_ylabel("uplift")
    ax_uplift.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax_uplift.grid(True, alpha=0.3)
    ax_uplift.legend(loc="upper right")
    ax_uplift.set_xlim(0.0, 1.0)

    fig.tight_layout()
    pdf_path = base_path / "classical_experiment_results.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    Path(f"classical_experiment_results_{flavor}.pdf").write_bytes(pdf_path.read_bytes())

    early_auc = _first_valid(auc_series)
    late_auc = _last_valid(auc_series)
    early_uplift_desc = _first_valid(uplift_b_series)
    late_uplift_desc = _last_valid(uplift_b_series)

    train_frac = report.dataset_summary["train_samples"] / report.dataset_summary["n_samples"]
    test_frac = report.dataset_summary["test_samples"] / report.dataset_summary["n_samples"]

    lines = [
        f"# ExtraBoost Classical Experiment ({flavor.upper()} loss)",
        "",
        "Synthetic dataset generated via the legacy boolean feature recipe (see `classical_test_approach.md`).",
        "",
        "## Configuration",
        "````json",
        json.dumps(asdict(report.config), indent=2),
        "````",
        "",
        "## Key Metrics",
        f"- overall ROC AUC: **{report.results.overall_auc:.3f}**",
    ]
    if early_auc:
        lines.append(f"- earliest evaluated window (t≈{early_auc[0]:.3f}) AUC: {early_auc[1]:.3f}")
    if late_auc:
        lines.append(f"- latest evaluated window (t≈{late_auc[0]:.3f}) AUC: {late_auc[1]:.3f}")
    if early_uplift_desc and late_uplift_desc:
        lines.append(
            f"- feature 1 (descending lift) uplift flips from {early_uplift_desc[1]:.3f} to {late_uplift_desc[1]:.3f}"
        )
    lines.append(
        f"- train/test split: {report.dataset_summary['train_samples']} ({train_frac:.2%}) / "
        f"{report.dataset_summary['test_samples']} ({test_frac:.2%}) samples"
    )
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `classical_experiment_results.pdf` — ROC AUC and uplift curves",
            "- `summary.json` — raw metrics dump",
        ]
    )

    md_path = base_path / "classical_experiment_results.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    Path(f"classical_experiment_results_{flavor}.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    base_dir = Path("reports/classical_experiment")
    base_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        n_samples=100_000,
        train_time_cutoff=0.4,
        uplift_feature_indices=(0, 1),
        auc_bin_edges=tuple(np.linspace(0.0, 1.0, 11)),
        evaluation_scope="dataset",
        learning_rate=0.2,
        max_depth=6,
        n_stages=250,
        reg_lambda=1e-4,
        threads=None,
    )

    configs = {
        "mse": ExperimentConfig(loss="mse", **common_kwargs),
        "logloss": ExperimentConfig(loss="logloss", **common_kwargs),
    }

    for flavor, cfg in configs.items():
        report = run_classical_pipeline(cfg)
        _write_report(base_dir / flavor, report, flavor)


if __name__ == "__main__":
    main()

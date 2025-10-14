from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

try:
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    PdfPages = None  # type: ignore[assignment]

from extra_boost_py.pipeline import (
    ExperimentConfig,
    PipelineReport,
    run_classical_pipeline,
)

REPORT_DIR = Path("reports/test_binary")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def plot_uplift(ax: plt.Axes, uplift_data: List[Tuple[Tuple[float, float], float]], title: str) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    centers = [0.5 * (interval[0] + interval[1]) for interval, _ in uplift_data]
    values = [val for _, val in uplift_data]
    ax.plot(centers, values, marker="o")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("uplift (mean_high - mean_low)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_auc(ax: plt.Axes, auc_data: List[Tuple[Tuple[float, float], float]]) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    centers = [0.5 * (interval[0] + interval[1]) for interval, _ in auc_data]
    values = [val for _, val in auc_data]
    ax.plot(centers, values, marker="o")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("ROC AUC across time bins")
    ax.grid(True, alpha=0.3)


def create_pdf_presentation(
    report: PipelineReport,
) -> None:
    if PdfPages is None:
        raise RuntimeError("matplotlib is required to create the PDF presentation.")
    pdf_path = REPORT_DIR / "experiment_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        config = report.config
        results = report.results
        dataset_summary = report.dataset_summary
        # slide 1
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        lines = [
            "Synthetic Dataset Experiment — Time-Dependent Boosting",
            "",
            "Dataset Parameters:",
            f"  Samples: {config.n_samples:,}",
            f"  Features: {dataset_summary['n_features']}",
            f"  Extra features: {dataset_summary['n_extra_features']}",
            f"  Train cutoff t< {config.train_time_cutoff:.2f}",
            "",
            "Booster Hyperparameters:",
            f"  Trees: {config.n_stages}",
            f"  Max depth: {config.max_depth}",
            f"  Learning rate: {config.learning_rate}",
            f"  Lambda: {config.reg_lambda}",
            f"  Loss: {config.loss}",
            f"  Threads: {config.threads}",
            "",
            "Outcomes:",
            f"  Overall test ROC AUC: {results.overall_auc:.4f}",
            f"  Example uplift (f0) early: {results.uplift_feature_0[3][1]:.4f}",
            f"  Example uplift (f0) late: {results.uplift_feature_0[-3][1]:.4f}",
        ]
        ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=16)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True)
        plot_uplift(axes[0], results.uplift_feature_0, "Uplift for f0")
        plot_uplift(axes[1], results.uplift_feature_10, "Uplift for f1")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        plot_auc(ax, results.time_slice_auc)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        info = pdf.infodict()
        info["Title"] = "Synthetic Dataset Boosting Report"
        info["Author"] = "extra_boost_py"


def write_markdown_report(report: PipelineReport, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Synthetic Dataset Experiment (Time-Dependent Boosting)\n\n")
        f.write("## Dataset Generation\n")
        f.write("```json\n")
        json.dump(report.dataset_summary, f, indent=2)
        f.write("\n```\n\n")

        f.write("## Booster Configuration\n")
        f.write("```json\n")
        json.dump(asdict(report.config), f, indent=2)
        f.write("\n```\n\n")

        f.write(f"**Overall ROC AUC**: {report.results.overall_auc:.4f}\n\n")

        def dump_table(title: str, data: List[Tuple[Tuple[float, float], float]]):
            f.write(f"### {title}\n")
            f.write("| Time interval | Value |\n")
            f.write("| --- | --- |\n")
            for (start, end), value in data:
                disp = "NaN" if math.isnan(value) else f"{value:.4f}"
                f.write(f"| [{start:.3f}, {end:.3f}) | {disp} |\n")
            f.write("\n")

        dump_table("ROC AUC by time interval", report.results.time_slice_auc)
        dump_table("Uplift for feature f0", report.results.uplift_feature_0)
        dump_table("Uplift for feature f10", report.results.uplift_feature_10)


def main() -> None:
    config = ExperimentConfig(
        n_samples=10_000,
        learning_rate=0.2,
        max_depth=6,
        n_stages=250,
        reg_lambda=1e-4,
        loss="mse",
        alpha=0.3,
        train_time_cutoff=0.4,
    )
    report = run_classical_pipeline(config)

    figures_dir = REPORT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    if plt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plot_uplift(axes[0], report.results.uplift_feature_0, "Uplift for f0 (initially correlated)")
        plot_uplift(axes[1], report.results.uplift_feature_10, "Uplift for f10 (initially anti-correlated)")
        plt.tight_layout()
        fig.savefig(figures_dir / "uplift_features.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_auc(ax, report.results.time_slice_auc)
        plt.tight_layout()
        fig.savefig(figures_dir / "roc_auc_time.png", dpi=200)
        plt.close(fig)

    if PdfPages is not None:
        create_pdf_presentation(report)

    write_markdown_report(report, REPORT_DIR / "report.md")

    summary_path = REPORT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(report.config),
                "dataset_summary": report.dataset_summary,
                "results": {
                    "overall_auc": report.results.overall_auc,
                    "time_slice_auc": report.results.time_slice_auc,
                    "uplift_feature_0": report.results.uplift_feature_0,
                    "uplift_feature_10": report.results.uplift_feature_10,
                },
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

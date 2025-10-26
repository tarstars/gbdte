from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np

from .booster import BoosterParams, ExtraBooster
from .classical_dataset import GeneratedDataset, generate_classical_dataset
from .go_lib import build_shared
from .metrics import roc_auc_score


@dataclass
class ExperimentConfig:
    """Configuration for the classical ExtraBoost binary scenario."""

    n_samples: int = 10_000
    learning_rate: float = 0.2
    max_depth: int = 6
    n_stages: int = 250
    reg_lambda: float = 1e-4
    loss: str = "mse"
    threads: int | None = None
    seed: int = 123
    alpha: float = 0.3
    train_time_cutoff: float | None = 0.4
    uplift_bins: int = 20
    auc_bins: int = 20
    uplift_min_count: int = 25
    auc_min_count: int = 50
    uplift_feature_indices: tuple[int, int] | None = (0, 1)
    auc_bin_edges: tuple[float, ...] | None = None
    evaluation_scope: str = "test"


@dataclass
class ExperimentResults:
    overall_auc: float
    time_slice_auc: List[Tuple[Tuple[float, float], float]]
    uplift_feature_0: List[Tuple[Tuple[float, float], float]]
    uplift_feature_10: List[Tuple[Tuple[float, float], float]]


@dataclass
class PipelineReport:
    config: ExperimentConfig
    dataset_summary: Dict[str, float]
    results: ExperimentResults


def _compute_uplift(
    feature_values: np.ndarray,
    target: np.ndarray,
    time: np.ndarray,
    *,
    bins: int,
    min_count: int,
) -> List[Tuple[Tuple[float, float], float]]:
    edges = np.linspace(time.min(), time.max(), bins + 1)
    median = np.median(feature_values)
    high_mask = feature_values > median
    low_mask = feature_values <= median
    results: List[Tuple[Tuple[float, float], float]] = []

    for left, right in zip(edges[:-1], edges[1:]):
        time_mask = (time >= left) & (time < right)
        mask_high = time_mask & high_mask
        mask_low = time_mask & low_mask
        if mask_high.sum() < min_count or mask_low.sum() < min_count:
            uplift = float("nan")
        else:
            uplift = float(target[mask_high].mean() - target[mask_low].mean())
        results.append(((float(left), float(right)), uplift))
    return results


def _evaluate_time_slices(
    time: np.ndarray,
    target: np.ndarray,
    preds: np.ndarray,
    *,
    bins: int,
    min_count: int,
    edges: np.ndarray | None = None,
) -> List[Tuple[Tuple[float, float], float]]:
    if edges is None:
        edges = np.linspace(time.min(), time.max(), bins + 1)
    else:
        edges = np.asarray(edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("auc_bin_edges must provide at least two sorted values.")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("auc_bin_edges must be strictly increasing.")
    results: List[Tuple[Tuple[float, float], float]] = []

    for left, right in zip(edges[:-1], edges[1:]):
        mask = (time >= left) & (time < right)
        if mask.sum() < min_count:
            auc = float("nan")
        else:
            try:
                auc = roc_auc_score(target[mask], preds[mask])
            except ValueError:
                auc = float("nan")
        results.append(((float(left), float(right)), auc))
    return results


def run_classical_pipeline(config: ExperimentConfig | None = None) -> PipelineReport:
    """Run dataset generation, ExtraBoost training, and time-based analysis."""

    build_shared()

    cfg = config or ExperimentConfig()
    threads = cfg.threads if cfg.threads is not None else os.cpu_count() or 4
    if cfg.threads is None:
        cfg = replace(cfg, threads=int(threads))

    dataset = generate_classical_dataset(
        n_samples=cfg.n_samples,
        alpha=cfg.alpha,
        seed=cfg.seed,
    )

    cutoff = cfg.train_time_cutoff if cfg.train_time_cutoff is not None else 0.4
    mask_train = dataset.time < cutoff
    mask_test = ~mask_train
    if mask_train.sum() == 0 or mask_test.sum() == 0:
        raise ValueError("Time cutoff results in empty train/test split.")
    train_ds = GeneratedDataset(
        dataset.features_inter[mask_train],
        dataset.features_extra[mask_train],
        dataset.target[mask_train],
        dataset.time[mask_train],
    )
    test_ds = GeneratedDataset(
        dataset.features_inter[mask_test],
        dataset.features_extra[mask_test],
        dataset.target[mask_test],
        dataset.time[mask_test],
    )

    params = BoosterParams(
        n_stages=cfg.n_stages,
        reg_lambda=cfg.reg_lambda,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        loss=cfg.loss,
        threads_num=threads,
    )

    monitor_datasets = [
        (train_ds.features_inter, train_ds.features_extra, train_ds.target, "train"),
        (test_ds.features_inter, test_ds.features_extra, test_ds.target, "test"),
    ]

    booster = ExtraBooster.train(
        train_ds.features_inter,
        train_ds.features_extra,
        train_ds.target,
        params=params,
        monitor_datasets=monitor_datasets,
    )

    predictions_test = booster.predict(test_ds.features_inter, test_ds.features_extra)

    if cfg.evaluation_scope == "test":
        eval_time = test_ds.time
        eval_target = test_ds.target
        eval_predictions = predictions_test
    elif cfg.evaluation_scope == "dataset":
        eval_time = dataset.time
        eval_target = dataset.target
        eval_predictions = booster.predict(dataset.features_inter, dataset.features_extra)
    else:
        raise ValueError(f"Unsupported evaluation_scope '{cfg.evaluation_scope}'.")

    overall_auc = roc_auc_score(eval_target, eval_predictions)

    bin_edges = (
        np.asarray(cfg.auc_bin_edges, dtype=np.float64) if cfg.auc_bin_edges is not None else None
    )

    time_slice_auc = _evaluate_time_slices(
        eval_time,
        eval_target,
        eval_predictions,
        bins=cfg.auc_bins,
        min_count=cfg.auc_min_count,
        edges=bin_edges,
    )
    feature_pair = cfg.uplift_feature_indices or (0, 1)
    idx_a, idx_b = feature_pair
    if idx_a >= dataset.features_inter.shape[1] or idx_b >= dataset.features_inter.shape[1]:
        raise IndexError("Configured uplift feature index exceeds dataset dimensionality.")

    uplift_f0 = _compute_uplift(
        dataset.features_inter[:, idx_a],
        dataset.target,
        dataset.time,
        bins=cfg.uplift_bins,
        min_count=cfg.uplift_min_count,
    )
    uplift_f10 = _compute_uplift(
        dataset.features_inter[:, idx_b],
        dataset.target,
        dataset.time,
        bins=cfg.uplift_bins,
        min_count=cfg.uplift_min_count,
    )

    dataset_summary = {
        "n_samples": cfg.n_samples,
        "n_features": int(dataset.features_inter.shape[1]),
        "n_extra_features": int(dataset.features_extra.shape[1]),
        "target_mean": float(dataset.target.mean()),
        "time_min": float(dataset.time.min()),
        "time_max": float(dataset.time.max()),
        "train_samples": int(train_ds.target.shape[0]),
        "test_samples": int(test_ds.target.shape[0]),
        "evaluation_scope": cfg.evaluation_scope,
    }

    booster.close()

    results = ExperimentResults(
        overall_auc=overall_auc,
        time_slice_auc=time_slice_auc,
        uplift_feature_0=uplift_f0,
        uplift_feature_10=uplift_f10,
    )

    return PipelineReport(config=cfg, dataset_summary=dataset_summary, results=results)


__all__ = [
    "ExperimentConfig",
    "ExperimentResults",
    "PipelineReport",
    "run_classical_pipeline",
]

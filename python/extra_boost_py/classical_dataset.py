from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class GeneratedDataset:
    """Container for the synthetic datasets produced by ExtraBoost helpers."""

    features_inter: np.ndarray
    features_extra: np.ndarray
    target: np.ndarray
    time: np.ndarray


@dataclass(frozen=True)
class ClassicalFeatureSpec:
    slope: float
    intercept: float
    beta: float


def generate_classical_dataset(
    n_samples: int,
    *,
    alpha: float = 0.3,
    seed: int | None = 42,
    ascend_spec: ClassicalFeatureSpec | None = None,
    descend_spec: ClassicalFeatureSpec | None = None,
) -> GeneratedDataset:
    """Recreates the legacy ExtraBoost synthetic dataset."""

    ascend = ascend_spec or ClassicalFeatureSpec(slope=0.75, intercept=0.5, beta=0.2)
    descend = descend_spec or ClassicalFeatureSpec(slope=-0.75, intercept=1.25, beta=0.1)

    rng = np.random.default_rng(seed)

    time = rng.random(n_samples)
    labels = (rng.random(n_samples) < alpha).astype(np.float64)

    feature_specs: list[ClassicalFeatureSpec] = [ascend, descend]
    for idx in range(15):
        feature_specs.append(descend if idx < 7 else ascend)

    features = np.empty((n_samples, len(feature_specs)), dtype=np.float64)
    denom = 1.0 / alpha - 1.0

    for col, spec in enumerate(feature_specs):
        lifts = spec.slope * time + spec.intercept
        gammas = ((1.0 / (alpha * lifts) - 1.0) / denom) * spec.beta
        probs = labels * spec.beta + (1.0 - labels) * gammas
        features[:, col] = (rng.random(n_samples) < probs).astype(np.float64)

    features_inter = np.ascontiguousarray(features, dtype=np.float64)
    time = np.ascontiguousarray(time, dtype=np.float64)
    labels = np.ascontiguousarray(labels, dtype=np.float64)
    features_extra = np.column_stack([np.ones_like(time), time]).astype(np.float64)

    return GeneratedDataset(features_inter, features_extra, labels, time)

def train_test_split(
    dataset: GeneratedDataset,
    *,
    ratio: float = 0.8,
    shuffle: bool = True,
    seed: int | None = 123,
) -> Tuple[GeneratedDataset, GeneratedDataset]:
    """Random train/test split helper for classical datasets."""

    if not (0.0 < ratio < 1.0):
        raise ValueError("ratio must lie in (0, 1).")

    n_samples = dataset.target.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    split = int(n_samples * ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Split resulted in an empty train or test set. Adjust ratio.")

    def _slice(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(arr[idx], dtype=np.float64)

    train = GeneratedDataset(
        features_inter=_slice(dataset.features_inter, train_idx),
        features_extra=_slice(dataset.features_extra, train_idx),
        target=_slice(dataset.target, train_idx),
        time=_slice(dataset.time, train_idx),
    )

    test = GeneratedDataset(
        features_inter=_slice(dataset.features_inter, test_idx),
        features_extra=_slice(dataset.features_extra, test_idx),
        target=_slice(dataset.target, test_idx),
        time=_slice(dataset.time, test_idx),
    )

    return train, test


__all__ = ["GeneratedDataset", "ClassicalFeatureSpec", "generate_classical_dataset", "train_test_split"]

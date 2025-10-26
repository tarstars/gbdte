"""Demonstrates the end-to-end workflow: dataset generation → Go booster → metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from extra_boost_py import (
    BoosterParams,
    ExtraBooster,
    build_shared,
    generate_classical_dataset,
    roc_auc_score,
    train_test_split,
)


def evaluate_time_slices(time: np.ndarray, target: np.ndarray, preds: np.ndarray, bins: int = 10) -> list[tuple[tuple[float, float], float]]:
    edges = np.linspace(time.min(), time.max(), bins + 1)
    results: list[tuple[tuple[float, float], float]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (time >= left) & (time < right)
        if mask.sum() < 3:
            continue
        auc = roc_auc_score(target[mask], preds[mask])
        results.append(((float(left), float(right)), auc))
    return results


def feature_correlation(dataset, feature_index: int, split: float = 0.9) -> tuple[float, float]:
    mask_early = dataset.time < split
    mask_late = dataset.time >= split
    feat = dataset.features_inter[:, feature_index]
    target = dataset.target

    def corr(m):
        if m.sum() < 3:
            return float("nan")
        return float(np.corrcoef(feat[m], target[m])[0, 1])

    return corr(mask_early), corr(mask_late)


def main() -> None:
    print("Building Go shared library …")
    build_shared()

    print("Generating synthetic dataset …")
    dataset = generate_classical_dataset(n_samples=2000, alpha=0.3)
    train_ds, test_ds = train_test_split(dataset, ratio=0.8)

    params = BoosterParams(n_stages=200, learning_rate=0.2, max_depth=6, loss="mse", threads_num=4)
    print("Training booster …")
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

    print("Scoring on the hold-out segment …")
    predictions = booster.predict(test_ds.features_inter, test_ds.features_extra)
    auc = roc_auc_score(test_ds.target, predictions)
    print(f"Overall ROC AUC on test set: {auc:.4f}")

    print("Time-sliced AUC (to display polarity shift):")
    for (start, end), slice_auc in evaluate_time_slices(test_ds.time, test_ds.target, predictions, bins=10):
        if np.isnan(slice_auc):
            status = "insufficient positives/negatives"
        else:
            status = f"AUC = {slice_auc:.4f}"
        print(f"  t in [{start:.2f}, {end:.2f}): {status}")

    early_corr, late_corr = feature_correlation(dataset, 0)
    early_corr2, late_corr2 = feature_correlation(dataset, 1)
    print(f"Feature f0 correlation with target: early={early_corr:.3f}, late={late_corr:.3f}.")
    print(f"Feature f1 correlation with target: early={early_corr2:.3f}, late={late_corr2:.3f}.")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "demo_model.ebm"
    curves_path = artifacts_dir / "demo_learning_curves.json"

    print(f"Saving model to {model_path} …")
    booster.save(model_path)
    booster.dump_learning_curves(curves_path)

    print("Reloading model and confirming predictions match …")
    reloaded = ExtraBooster.load(model_path)
    preds_reloaded = reloaded.predict(test_ds.features_inter, test_ds.features_extra)
    diff = np.max(np.abs(predictions - preds_reloaded))
    print(f"Max abs difference between original and reloaded predictions: {diff:.3e}")

    booster.close()
    reloaded.close()


if __name__ == "__main__":
    main()

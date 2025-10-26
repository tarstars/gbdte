#!/usr/bin/env python3
"""Analyse the filtered component-only dataset.

Computes least-squares weights for the (1, t, sin(k t), cos(k t)) basis and reports RMSE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[3] / "datasets/mse/component_f000.csv"

def main() -> None:
    df = pd.read_csv(DATA_PATH)
    features = df[["e_1", "e_2", "e_3", "e_4"]].to_numpy(dtype=float)
    target = df["y"].to_numpy(dtype=float)

    coeffs, residuals, rank, singular_vals = np.linalg.lstsq(features, target, rcond=None)
    predictions = features @ coeffs
    rmse = np.sqrt(np.mean((target - predictions) ** 2))

    print(f"Dataset path: {DATA_PATH}")
    print(f"Rows: {len(df)}, Rank: {rank}")
    print("Coefficients (bias, t, sin, cos):")
    for idx, value in enumerate(coeffs):
        print(f"  w{idx}: {value:.9f}")
    print(f"RMSE: {rmse:.9e}")
    if residuals.size:
        print(f"Residual sum of squares: {residuals[0]:.9e}")

if __name__ == "__main__":
    main()

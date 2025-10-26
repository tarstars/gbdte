#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an 'extra boosting' dataset from a dotted-letter photo.

Pipeline
1) Dot extraction (Otsu threshold + connected components)
2) RAW coordinates: x in [0,1] (divide by width), y scaled by width, origin bottom-left
3) Signal filter in RAW: keep ymin < y < ymax
4) Normalize (uniform scale; horizontal span=1; shift so left-bottom = (0,0))
5) Greedy max-consensus with fixed k:
     y = A + B t + C sin(k t) + D cos(k t)
   tolerance ±tol, N iterations; remove captured each round
6) Build final dataset with columns:
   f_1,f_2,f_3,e_1,e_2,e_3,e_4,t,y
   where (f_1,f_2,f_3) is the 3-bit code of (component_iter-1),
   e_1=1, e_2=t, e_3=sin(t), e_4=cos(t)

Usage:
  python make_extra_boosting_dataset.py --image INPUT.png --k 50 --iters 8 --tol 0.01 --ymin 0.2 --ymax 0.6 --outdir out
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi


# ---------- utilities ----------
def otsu_threshold(gray_u8: np.ndarray) -> int:
    """Classic Otsu for uint8 grayscale."""
    hist, _ = np.histogram(gray_u8, bins=256, range=(0, 255))
    total = gray_u8.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0.0
    max_var = -1.0
    threshold = 127
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return int(threshold)


def extract_dots(image_path: str, min_area=10, max_area=1500, thresh_bias=-10):
    """Return raw (x,y) points with x in [0,1], y scaled by width, origin bottom-left."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    H, W = arr.shape
    th = max(0, otsu_threshold(arr) + thresh_bias)
    mask = arr < th
    # clean borders
    mask[0, :] = mask[-1, :] = False
    mask[:, 0] = mask[:, -1] = False

    labels, num = ndi.label(mask)
    if num == 0:
        return pd.DataFrame(columns=["x", "y"])
    coms = ndi.center_of_mass(mask, labels=labels, index=range(1, num + 1))
    areas = np.bincount(labels.ravel())
    centroids = []
    for idx, (cy, cx) in enumerate(coms, start=1):
        area = areas[idx] if idx < len(areas) else 0
        if min_area <= area <= max_area:
            centroids.append((cx, cy))
    if not centroids:
        return pd.DataFrame(columns=["x", "y"])
    xs, ys = np.array(centroids).T
    x_raw = xs / W
    y_raw = (H - ys) / W  # scale by width; origin bottom-left
    df = pd.DataFrame({"x": x_raw, "y": y_raw}).sort_values("x").reset_index(drop=True)
    return df


def normalize_signal(df_raw: pd.DataFrame, ymin=0.2, ymax=0.6):
    """Filter to (ymin,ymax) in RAW space; then normalize with uniform scale so x-span=1 and left-bottom at (0,0)."""
    sig = df_raw[(df_raw["y"] > ymin) & (df_raw["y"] < ymax)].copy()
    if len(sig) == 0:
        return sig
    x = sig["x"].to_numpy()
    y = sig["y"].to_numpy()
    xmin, xmax = float(x.min()), float(x.max())
    scale = 1.0 / (xmax - xmin + 1e-12)
    x_scaled = (x - xmin) * scale
    # uniform scale applied to y as well; then shift so min y -> 0
    y_scaled = (y - y.min()) * scale
    # final left-bottom shift (should already be 0,0 but keep for robustness)
    idx_lb = np.lexsort((y_scaled, x_scaled))[0]
    x_norm = x_scaled - x_scaled[idx_lb]
    y_norm = y_scaled - y_scaled[idx_lb]
    return pd.DataFrame({"x": x_norm, "y": y_norm})


def ransac_greedy_fixed_k(t, y, k, tol, n_iter=25000, random_state=None):
    """Return (A,B,C,D), mask maximizing inliers within tol for fixed k, with polish refit on inliers."""
    rng = np.random.default_rng(random_state)
    n = len(y)
    ones = np.ones_like(t)
    s = np.sin(k * t)
    c = np.cos(k * t)
    X = np.column_stack([ones, t, s, c])

    best_cnt, best_beta, best_mask = -1, None, None
    trials = min(n_iter, max(1, n * 120))
    for _ in range(trials):
        if n < 4:
            break
        idx = rng.choice(n, size=4, replace=False)
        beta, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        yhat = X @ beta
        mask = np.abs(y - yhat) <= tol
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_beta, best_mask = cnt, beta, mask

    # polish/refit on inliers
    if best_beta is not None and best_cnt >= 4:
        beta_ref, *_ = np.linalg.lstsq(X[best_mask], y[best_mask], rcond=None)
        yhat = X @ beta_ref
        mask = np.abs(y - yhat) <= tol
        cnt = int(mask.sum())
        if cnt >= best_cnt:
            best_beta, best_mask, best_cnt = beta_ref, mask, cnt

    return best_beta, best_mask, best_cnt


def make_dataset_from_components(components, k, tol):
    """
    components: list of dicts with keys:
        'iter' (1-based), 'A','B','C','D', 'points' (DataFrame with x,y and 'captured' boolean)
    Returns DataFrame with f_1..f_3, e_1..e_4, t, y
    """
    rows = []
    for comp in components:
        it = comp["iter"]
        A, B, C, D = comp["A"], comp["B"], comp["C"], comp["D"]
        pts = comp["points"]
        cap = pts[pts["captured"]].copy()
        if cap.empty:
            continue
        t = cap["x"].to_numpy()
        y_model = A + B * t + C * np.sin(k * t) + D * np.cos(k * t)

        # binary-encode (it-1) into 3 bits: f1 MSB, f3 LSB
        n = np.clip(np.array(it - 1, dtype=int), 0, 7)
        f1 = (n >> 2) & 1
        f2 = (n >> 1) & 1
        f3 = n & 1

        out = pd.DataFrame(
            {
                "f_1": f1,
                "f_2": f2,
                "f_3": f3,
                "e_1": 1.0,
                "e_2": t,
                "e_3": np.sin(k * t),
                "e_4": np.cos(k * t),
                "t": t,
                "y": y_model,
            }
        )
        rows.append(out)

    if rows:
        return pd.concat(rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=["f_1","f_2","f_3","e_1","e_2","e_3","e_4","t","y"])


# ---------- main ----------
def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) dots -> RAW points
    df_raw = extract_dots(
        args.image, min_area=args.min_area, max_area=args.max_area, thresh_bias=args.thresh_bias
    )
    raw_csv = outdir / "points_raw.csv"
    df_raw.to_csv(raw_csv, index=False)

    # 2) normalize signal
    df_norm = normalize_signal(df_raw, ymin=args.ymin, ymax=args.ymax)
    norm_csv = outdir / "signal_normalized.csv"
    df_norm.to_csv(norm_csv, index=False)

    # 3) greedy fixed-k iterations
    remaining = df_norm.copy().reset_index(drop=True)
    components = []
    for it in range(1, args.iters + 1):
        if len(remaining) < 4:
            break
        t = remaining["x"].to_numpy()
        y = remaining["y"].to_numpy()
        beta, mask, cnt = ransac_greedy_fixed_k(
            t, y, args.k, args.tol, n_iter=args.ransac_trials, random_state=args.seed + it
        )
        if beta is None:
            break
        A, B, C, D = map(float, beta)
        rep = {
            "iteration": it,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "k": float(args.k),
            "tol": float(args.tol),
            "remaining_before": int(len(remaining)),
            "captured_iter": int(cnt),
        }
        # save per-iteration report & points
        (outdir / f"iter_{it:02d}_report.json").write_text(json.dumps(rep, indent=2))
        pts = remaining.copy()
        pts["captured"] = mask
        pts.to_csv(outdir / f"iter_{it:02d}_points.csv", index=False)

        components.append({"iter": it, "A": A, "B": B, "C": C, "D": D, "points": pts})

        # remove captured for next round
        remaining = remaining[~mask].reset_index(drop=True)

    # 4) final dataset
    ds = make_dataset_from_components(components, k=args.k, tol=args.tol)
    ds_csv = outdir / "extra_boosting_static_features.csv"
    ds.to_csv(ds_csv, index=False)

    # small summary
    per_iter = [{"iter": c["iter"], "captured": int(c["points"]["captured"].sum())} for c in components]
    print("\nPer-iteration captures:", per_iter)
    print(f"\nSaved:\n  RAW points -> {raw_csv}\n  Normalized signal -> {norm_csv}\n  Dataset -> {ds_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create an extra boosting dataset from a dotted-letter photo.")
    p.add_argument("--image", required=True, help="Path to input image (png/jpg).")
    p.add_argument("--outdir", default="out", help="Output directory.")
    p.add_argument("--k", type=float, default=50.0, help="Fixed frequency for the basis.")
    p.add_argument("--iters", type=int, default=8, help="Number of greedy iterations.")
    p.add_argument("--tol", type=float, default=0.01, help="Tolerance for capture band.")
    p.add_argument("--ymin", type=float, default=0.2, help="Signal filter lower bound in RAW y.")
    p.add_argument("--ymax", type=float, default=0.6, help="Signal filter upper bound in RAW y.")
    p.add_argument("--ransac-trials", type=int, default=30000, help="RANSAC/greedy random trials per iteration.")
    p.add_argument("--seed", type=int, default=1234, help="Random seed base.")
    # dot extraction knobs (generally fine as defaults)
    p.add_argument("--min-area", type=int, default=10, help="Min CC area to keep (pixels).")
    p.add_argument("--max-area", type=int, default=1500, help="Max CC area to keep (pixels).")
    p.add_argument("--thresh-bias", type=int, default=-10, help="Bias added to Otsu threshold (negative makes stricter).")

    args = p.parse_args()
    main(args)

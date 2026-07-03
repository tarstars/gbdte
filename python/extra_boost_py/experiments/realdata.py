"""E3: real datasets with temporal splits, loaded into the standard Bench form.

Raw data comes from Kaggle (credentials required); loaders download on first use into
datasets/realdata/ (gitignored) and cache a compact parquet extract. The transform is
pure and unit-tested offline: numeric features kept, categoricals ordinal-encoded,
feature NaNs median/`-1` imputed (no target involved), time normalized to [0,1],
temporal split at the 0.75 quantile.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .benchgen import Bench


@dataclass(frozen=True)
class RealDatasetSpec:
    name: str
    kaggle_ref: str
    kaggle_kind: str          # "dataset" | "competition"
    files: Tuple[str, ...]
    time_col: str
    target_col: str
    task: str                 # "mse" | "logloss"


REAL_DATASETS = {
    "weather": RealDatasetSpec(
        name="weather", kaggle_ref="pcovkrd84mejm/tabred-weather",
        kaggle_kind="dataset", files=("weather.parquet",),
        time_col="fact_time", target_col="fact_temperature", task="mse"),
    "sberbank": RealDatasetSpec(
        name="sberbank", kaggle_ref="sberbank-russian-housing-market",
        kaggle_kind="competition", files=("train.csv.zip",),
        time_col="timestamp", target_col="price_doc", task="mse"),
    "homecredit": RealDatasetSpec(
        name="homecredit", kaggle_ref="home-credit-credit-risk-model-stability",
        kaggle_kind="competition",
        files=("csv_files/train/train_base.csv",
               "csv_files/train/train_static_0_0.csv",
               "csv_files/train/train_static_0_1.csv",
               "csv_files/train/train_static_cb_0.csv"),
        time_col="date_decision", target_col="target", task="logloss"),
}


def frame_to_bench(df: pd.DataFrame, spec: RealDatasetSpec, seed: int,
                   n_max: int = 60000) -> Bench:
    df = df.dropna(subset=[spec.time_col, spec.target_col])
    if len(df) > n_max:
        rng = np.random.default_rng(seed)
        df = df.iloc[rng.choice(len(df), n_max, replace=False)]
    t_raw = pd.to_datetime(df[spec.time_col]).astype("int64").to_numpy(dtype=np.float64)
    t = (t_raw - t_raw.min()) / max(t_raw.max() - t_raw.min(), 1.0)

    out = {"t": t, "e_0": np.ones(len(df)), "e_1": t,
           "y": df[spec.target_col].to_numpy(dtype=np.float64)}
    partition_cols = []
    for col in df.columns:
        if col in (spec.time_col, spec.target_col):
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            v = s.to_numpy(dtype=np.float64)
            med = np.nanmedian(v)
            out[col] = np.where(np.isnan(v), 0.0 if np.isnan(med) else med, v)
        else:
            codes = s.astype("category").cat.codes.to_numpy(dtype=np.float64)
            out[col] = codes  # NaN becomes -1: its own category
        partition_cols.append(col)

    frame = pd.DataFrame(out).sort_values("t").reset_index(drop=True)
    cut = float(frame["t"].quantile(0.75))
    return Bench(df=frame, partition_cols=partition_cols,
                 extra_cols=["e_0", "e_1"], task=spec.task, cut=cut)


def _download(spec: RealDatasetSpec, cache_dir: Path) -> Path:
    raw = cache_dir / spec.name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    kind = "datasets" if spec.kaggle_kind == "dataset" else "competitions"
    for f in spec.files:
        target = raw / Path(f).name
        stem = target.with_suffix("") if target.suffix == ".zip" else target
        if target.exists() or stem.exists():
            continue
        cmd = ["kaggle", kind, "download", spec.kaggle_ref, "-f", f, "-p", str(raw)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"kaggle download failed for {spec.name}: {res.stderr.strip()}\n"
                f"run manually: {' '.join(cmd)}")
    for z in raw.glob("*.zip"):
        subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(raw)], check=True)
        z.unlink()
    return raw


def _read_raw(spec: RealDatasetSpec, raw: Path) -> pd.DataFrame:
    if spec.name == "weather":
        return pd.read_parquet(raw / "weather.parquet")
    if spec.name == "sberbank":
        return pd.read_csv(raw / "train.csv")
    if spec.name == "homecredit":
        base = pd.read_csv(raw / "train_base.csv")
        for name in ("train_static_0_0.csv", "train_static_0_1.csv",
                     "train_static_cb_0.csv"):
            part = pd.read_csv(raw / name)
            part = part.drop_duplicates("case_id")
            base = base.merge(part, on="case_id", how="left",
                              suffixes=("", f"_{name.split('.')[0]}"))
        return base.drop(columns=["case_id", "MONTH", "WEEK_NUM"], errors="ignore")
    raise KeyError(spec.name)


def load_real(name: str, seed: int = 0, n_max: int = 60000,
              cache_dir: Path = Path("datasets/realdata")) -> Bench:
    spec = REAL_DATASETS[name]
    cache = Path(cache_dir) / name / "extract.parquet"
    if not cache.exists():
        raw = _download(spec, Path(cache_dir))
        df = _read_raw(spec, raw)
        # compact cache: pre-subsample very large frames deterministically
        if len(df) > 400000:
            df = df.iloc[np.random.default_rng(0).choice(len(df), 400000, replace=False)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    df = pd.read_parquet(cache)
    return frame_to_bench(df, spec, seed=seed, n_max=n_max)


__all__ = ["RealDatasetSpec", "REAL_DATASETS", "frame_to_bench", "load_real"]

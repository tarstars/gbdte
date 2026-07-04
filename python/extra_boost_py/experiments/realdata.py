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
    drop_cols: Tuple[str, ...] = ()   # ids/pseudo-time columns excluded from features
    log_target: bool = False          # y -> log1p(y) (multiplicative/rate targets)


REAL_DATASETS = {
    "weather": RealDatasetSpec(
        name="weather", kaggle_ref="pcovkrd84mejm/tabred-weather",
        kaggle_kind="dataset", files=("weather.parquet",),
        time_col="fact_time", target_col="fact_temperature", task="mse"),
    "sberbank": RealDatasetSpec(
        name="sberbank", kaggle_ref="sberbank-russian-housing-market",
        kaggle_kind="competition", files=("train.csv.zip",),
        time_col="timestamp", target_col="price_doc", task="mse",
        drop_cols=("id",)),   # monotone row index = pseudo-time, drift score 1.0
    "owid_childmort": RealDatasetSpec(
        name="owid_childmort", kaggle_ref="child-mortality|child_mortality_rate",
        kaggle_kind="owid", files=(), time_col="year", target_col="y", task="mse",
        log_target=True),   # mortality declines exponentially -> log ~linear in year
    "gassensor": RealDatasetSpec(
        name="gassensor",
        kaggle_ref="https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
        kaggle_kind="uci", files=("gas.zip",),
        time_col="batch", target_col="y", task="logloss",
        drop_cols=("gas",)),   # raw class kept only to build the binary target
    "latam": RealDatasetSpec(
        name="latam",
        kaggle_ref="mendeley:mhb4zn3258",   # public direct download, not kaggle
        kaggle_kind="mendeley",
        files=("customer_data.csv", "transactions_data.csv"),
        time_col="month", target_col="y_log_count", task="mse",
        drop_cols=("customer_id",)),
    "homecredit": RealDatasetSpec(
        name="homecredit", kaggle_ref="home-credit-credit-risk-model-stability",
        kaggle_kind="competition",
        files=("csv_files/train/train_base.csv",
               "csv_files/train/train_static_0_0.csv",
               "csv_files/train/train_static_0_1.csv",
               "csv_files/train/train_static_cb_0.csv"),
        time_col="date_decision", target_col="target", task="logloss"),
}


_LATAM_STATIC_COLS = (
    "age", "gender", "location", "income_bracket", "occupation", "education_level",
    "marital_status", "household_size", "acquisition_channel", "customer_segment",
    "savings_account", "credit_card", "personal_loan", "investment_account",
    "insurance_product", "active_products",
)   # everything else in the card is a whole-2023 aggregate -> temporal leakage


def _latam_frame(card: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Client x month grid with y = log1p(monthly transaction count).

    Zero-activity months are materialized: churn shows up as trailing zeros, which is
    exactly the drift signal the benchmark is about."""
    months = pd.date_range("2023-01-01", periods=12, freq="MS").strftime("%Y-%m-%d")
    tx = tx.copy()
    tx["month"] = pd.to_datetime(tx["date"]).dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d")
    counts = tx.groupby(["customer_id", "month"]).size().rename("n").reset_index()

    grid = pd.MultiIndex.from_product(
        [card["customer_id"].unique(), months], names=["customer_id", "month"]
    ).to_frame(index=False)
    grid = grid.merge(counts, on=["customer_id", "month"], how="left")
    grid["y_log_count"] = np.log1p(grid["n"].fillna(0.0))
    grid = grid.drop(columns=["n"])

    static = card[["customer_id", *_LATAM_STATIC_COLS]]
    return grid.merge(static, on="customer_id", how="left")


def frame_to_bench(df: pd.DataFrame, spec: RealDatasetSpec, seed: int,
                   n_max: int = 60000) -> Bench:
    df = df.dropna(subset=[spec.time_col, spec.target_col])
    if len(df) > n_max:
        rng = np.random.default_rng(seed)
        df = df.iloc[rng.choice(len(df), n_max, replace=False)]
    time_series = df[spec.time_col]
    if pd.api.types.is_numeric_dtype(time_series):
        t_raw = time_series.to_numpy(dtype=np.float64)   # batch index, year, etc.
    else:
        t_raw = pd.to_datetime(time_series).astype("int64").to_numpy(dtype=np.float64)
    t = (t_raw - t_raw.min()) / max(t_raw.max() - t_raw.min(), 1.0)

    y_raw = df[spec.target_col].to_numpy(dtype=np.float64)
    if spec.log_target:
        y_raw = np.log1p(np.clip(y_raw, 0.0, None))
    out = {"t": t, "e_0": np.ones(len(df)), "e_1": t, "y": y_raw}
    partition_cols = []
    for col in df.columns:
        if col in (spec.time_col, spec.target_col) or col in spec.drop_cols:
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


def _parse_gas_batch(path, batch_idx: int) -> list:
    """Parse a UCI Gas Sensor Drift batch*.dat line. Robust to both the real layout
    'label idx:val ...' and the documented 'label;conc idx:val ...' (concentration, if
    present, is embedded in the first token and dropped)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            feats = {"gas": int(float(toks[0].split(";")[0])), "batch": batch_idx}
            for tok in toks[1:]:
                if ":" in tok:
                    idx, val = tok.split(":")
                    feats[f"s{int(idx)}"] = float(val)
            rows.append(feats)
    return rows


def _worldbank_frame(values: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Country-year panel with country identity WITHHELD (group must be inferred from
    region/income/initial-level). y = indicator value; init_value = per-country value at
    its earliest available year."""
    values = values.dropna(subset=["value"]).sort_values(["iso3", "year"])
    init = values.groupby("iso3")["value"].first().rename("init_value")
    df = values.merge(init, on="iso3").merge(meta, on="iso3", how="left")
    df["y"] = df["value"].to_numpy(dtype=np.float64)
    df["region"] = df["region"].astype("category").cat.codes.astype(float)
    df["income"] = df["income"].astype("category").cat.codes.astype(float)
    return df[["region", "income", "init_value", "year", "y"]].reset_index(drop=True)


_OWID_CONTINENTS = ("https://ourworldindata.org/grapher/"
                    "continents-according-to-our-world-in-data.csv?csvType=full")


def _owid_csv(url: str) -> pd.DataFrame:
    import io
    import urllib.request
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return pd.read_csv(io.BytesIO(r.read()))
        except Exception:
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


def _fetch_owid(spec: RealDatasetSpec) -> pd.DataFrame:
    """Country-year panel from Our World in Data. Country identity is withheld; the group
    must be inferred from continent (region) + initial indicator level."""
    slug, value_col = spec.kaggle_ref.split("|")
    ind = _owid_csv(f"https://ourworldindata.org/grapher/{slug}.csv"
                    f"?csvType=full&useColumnShortNames=true")
    cont = _owid_csv(_OWID_CONTINENTS)
    cont = cont.rename(columns={cont.columns[-1]: "region"})[["Code", "region"]]
    values = ind.rename(columns={"code": "iso3", value_col: "value"})
    values = values[values["iso3"].notna() & (values["iso3"] != "")]
    values = values[["iso3", "year", "value"]]
    meta = cont.rename(columns={"Code": "iso3"}).drop_duplicates("iso3")
    meta["income"] = "NA"                       # placeholder; region + init_value carry it
    values = values[values["iso3"].isin(meta["iso3"])]
    return _worldbank_frame(values, meta)


def _download(spec: RealDatasetSpec, cache_dir: Path) -> Path:
    raw = cache_dir / spec.name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if spec.kaggle_kind == "uci":
        has_dat = any(raw.rglob("batch*.dat"))
        if not has_dat:
            zip_path = raw / "gas.zip"
            subprocess.run(["curl", "-sL", "--max-time", "600", "-o", str(zip_path),
                            spec.kaggle_ref], check=True)
            subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(raw)], check=True)
        return raw
    if spec.kaggle_kind == "owid":
        return raw   # fetched in _read_raw
    if spec.kaggle_kind == "mendeley":
        for f in spec.files:
            target = raw / f
            if target.exists():
                continue
            res = subprocess.run(["curl", "-sL", "--max-time", "900",
                                  "-o", str(target), _LATAM_URLS[f]],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"mendeley download failed for {f}: {res.stderr}")
        return raw
    kind = "datasets" if spec.kaggle_kind == "dataset" else "competitions"
    for f in spec.files:
        target = raw / Path(f).name
        stem = target.with_suffix("") if target.suffix == ".zip" else target
        if target.exists() or stem.exists():
            continue
        import sys
        kaggle_bin = str(Path(sys.executable).parent / "kaggle")
        cmd = [kaggle_bin, kind, "download", spec.kaggle_ref, "-f", f, "-p", str(raw)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"kaggle download failed for {spec.name}: {res.stderr.strip()}\n"
                f"run manually: {' '.join(cmd)}")
    for z in raw.glob("*.zip"):
        subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(raw)], check=True)
        z.unlink()
    return raw


_LATAM_URLS = {
    "customer_data.csv": "https://data.mendeley.com/public-files/datasets/mhb4zn3258/files/0bd16d9c-55c4-4795-8545-db6b1de1f7fc/file_downloaded",
    "transactions_data.csv": "https://data.mendeley.com/public-files/datasets/mhb4zn3258/files/2afa7996-75b3-42ef-9c39-ed5e028d49f9/file_downloaded",
}


def _read_raw(spec: RealDatasetSpec, raw: Path) -> pd.DataFrame:
    if spec.name == "latam":
        card = pd.read_csv(raw / "customer_data.csv")
        tx = pd.read_csv(raw / "transactions_data.csv")
        return _latam_frame(card, tx)
    if spec.kaggle_kind == "owid":
        return _fetch_owid(spec)
    if spec.name == "gassensor":
        import glob
        dat_files = sorted(glob.glob(str(raw / "**" / "batch*.dat"), recursive=True))
        rows = []
        for path in dat_files:
            b = int("".join(ch for ch in Path(path).stem if ch.isdigit()))
            rows.extend(_parse_gas_batch(path, b))
        df = pd.DataFrame(rows).fillna(0.0)
        df["y"] = (df["gas"] == 1).astype(float)   # one-vs-rest, fixed class 1 (Ethanol)
        return df
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

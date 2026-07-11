import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

from ..config import HDF5_PATH, PARQUET_DIR


def load_gauge_info() -> pd.DataFrame:
    try:
        df = pd.read_hdf(HDF5_PATH, "table_info")
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(f"Failed to load gauge info from {HDF5_PATH}: {e}")
    df["gauge_code"] = df["gauge_code"].astype(str)
    return df


def load_rainfall_data(
    gauge_codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
    chunksize: int = 0,
) -> pd.DataFrame:
    if sample is not None:
        codes_df = pd.read_hdf(HDF5_PATH, "table_data", columns=["gauge_code"])
        unique_codes = codes_df["gauge_code"].unique().astype(str)
        rng = np.random.default_rng(seed)
        sampled = rng.choice(unique_codes, min(sample, len(unique_codes)), replace=False)
        query_codes = set(sampled)
        if gauge_codes is not None:
            query_codes &= set(gauge_codes)
        where = "(" + " | ".join([f"gauge_code == '{c}'" for c in query_codes]) + ")"
        try:
            df = pd.read_hdf(HDF5_PATH, "table_data", where=where)
        except (FileNotFoundError, KeyError) as e:
            raise FileNotFoundError(f"Failed to load rainfall data from {HDF5_PATH}: {e}")
        df["gauge_code"] = df["gauge_code"].astype(str)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    if chunksize > 0:
        pieces = []
        for chunk in pd.read_hdf(HDF5_PATH, "table_data", iterator=True, chunksize=chunksize):
            chunk["gauge_code"] = chunk["gauge_code"].astype(str)
            chunk["datetime"] = pd.to_datetime(chunk["datetime"])
            if gauge_codes is not None:
                chunk = chunk[chunk["gauge_code"].isin(gauge_codes)]
            if start_date is not None:
                chunk = chunk[chunk["datetime"] >= start_date]
            if end_date is not None:
                chunk = chunk[chunk["datetime"] <= end_date]
            pieces.append(chunk)
        if not pieces:
            return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True)

    try:
        df = pd.read_hdf(HDF5_PATH, "table_data")
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(f"Failed to load rainfall data from {HDF5_PATH}: {e}")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if gauge_codes is not None:
        df = df[df["gauge_code"].isin(gauge_codes)]
    if start_date is not None:
        df = df[df["datetime"] >= start_date]
    if end_date is not None:
        df = df[df["datetime"] <= end_date]
    return df


def load_daily_data() -> pd.DataFrame:
    try:
        df = pd.read_hdf(HDF5_PATH, "table_data_daily")
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(f"Failed to load daily data from {HDF5_PATH}: {e}")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_daily_hq_data() -> pd.DataFrame:
    try:
        df = pd.read_hdf(HDF5_PATH, "table_data_daily_hq")
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(f"Failed to load daily HQ data from {HDF5_PATH}: {e}")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_monthly_hq_data() -> pd.DataFrame:
    try:
        df = pd.read_hdf(HDF5_PATH, "table_data_monthly_hq")
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(f"Failed to load monthly HQ data from {HDF5_PATH}: {e}")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_gauge_data_by_code(df_rainfall: pd.DataFrame, code: str) -> pd.DataFrame:
    filt = df_rainfall["gauge_code"] == code
    temp = df_rainfall[filt].copy()
    temp.sort_values("datetime", ascending=True, inplace=True)
    temp.reset_index(drop=True, inplace=True)
    return temp


def get_stations_by_network(gauge_info: pd.DataFrame, network: str) -> List[str]:
    filt = gauge_info["network"] == network
    return list(gauge_info[filt]["gauge_code"].unique())


# ---- Parquet-compatible loaders ----


def _check_parquet_available() -> bool:
    return PARQUET_DIR.exists() and any(PARQUET_DIR.glob("*.parquet"))


def load_gauge_info_parquet() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_DIR / "table_info.parquet")
    df["gauge_code"] = df["gauge_code"].astype(str)
    return df


def load_rainfall_data_parquet(
    gauge_codes: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    sample: Optional[int] = None,
) -> pd.DataFrame:
    files = list(PARQUET_DIR.glob("table_data/**/*.parquet"))

    if states:
        files = [f for f in files if any(f"state={s}" in str(f.parent) for s in states)]
    if years:
        files = [f for f in files if any(f"year={y}" in f.stem for y in years)]

    pieces = []
    for f in files:
        df = pd.read_parquet(f, columns=["gauge_code", "datetime", "rain_mm"])
        pieces.append(df)

    df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if df.empty:
        return df

    df["gauge_code"] = df["gauge_code"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"])

    if gauge_codes is not None:
        df = df[df["gauge_code"].isin(gauge_codes)]
    if sample is not None:
        codes = df["gauge_code"].unique()
        sampled = np.random.choice(codes, min(sample, len(codes)), replace=False)
        df = df[df["gauge_code"].isin(sampled)]

    return df


def load_daily_data_parquet() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_DIR / "table_data_daily.parquet")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df

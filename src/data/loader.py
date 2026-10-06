"""
Data loading utilities for QC-VORONOI.
Supports reading directly from UNIPLU partitioned ZIP archives (containing Parquet files),
standalone Parquet files, or legacy HDF5 files.
"""

import io
import re
import zipfile
from pathlib import Path
from typing import Optional, List, Set, Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import HDF5_PATH, PARQUET_DIR, UNIPLU_DIR


# ─── Internal helpers for UNIPLU ZIPs ───────────────────────────────

def _is_uniplu_available() -> bool:
    """Check if UNIPLU zip directory exists and contains files."""
    return UNIPLU_DIR.exists() and any(UNIPLU_DIR.glob("*.zip"))


def _scan_uniplu_zips() -> List[Tuple[str, int, Path]]:
    """Return list of (state, year, zip_path) found in UNIPLU_DIR."""
    if not UNIPLU_DIR.exists():
        return []

    pattern = re.compile(r"^([A-Z]{2})_(\d{4})\.zip$")
    results = []
    for p in UNIPLU_DIR.glob("*.zip"):
        m = pattern.match(p.name)
        if m:
            state, year = m.group(1), int(m.group(2))
            results.append((state, year, p))
    return results


def _filter_uniplu_zips(
    states: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
) -> List[Path]:
    """Filter zip files by optional state list and year list."""
    scanned = _scan_uniplu_zips()
    states_set = {s.upper() for s in states} if states else None
    years_set = set(years) if years else None

    selected = []
    for st, yr, p in scanned:
        if states_set and st not in states_set:
            continue
        if years_set and yr not in years_set:
            continue
        selected.append(p)
    return sorted(selected)


# ─── Public Loaders: Gauge Info ──────────────────────────────────────

def load_gauge_info(
    states: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load metadata of rainfall gauges.

    Prioritizes reading directly from UNIPLU/*.zip archives if present.
    Falls back to legacy HDF5 if UNIPLU is not present.

    Parameters
    ----------
    states : list of str, optional
        Filter gauges by states/UFs (e.g. ['SP', 'RJ']).
    years : list of int, optional
        Only inspect archives corresponding to these years.
    use_cache : bool, default True
        When True and no state/year filter is specified, uses/saves a cache file
        for instant loading.
    """
    if _is_uniplu_available():
        cache_file = UNIPLU_DIR / ".cache_table_info.parquet"

        # Fast path: load cached consolidated table_info if no specific filters applied
        if use_cache and states is None and years is None and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df["gauge_code"] = df["gauge_code"].astype(str)
                return df
            except Exception:
                pass

        zip_paths = _filter_uniplu_zips(states=states, years=years)
        if not zip_paths:
            return pd.DataFrame(
                columns=[
                    "gauge_code", "city", "state", "lat", "long",
                    "time_step", "elevation", "UTC", "network", "responsible"
                ]
            )

        dfs = []
        for zp in zip_paths:
            try:
                with zipfile.ZipFile(zp) as zf:
                    if "table_info.parquet" in zf.namelist():
                        data = zf.read("table_info.parquet")
                        dfs.append(pd.read_parquet(io.BytesIO(data)))
            except Exception:
                continue

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df["gauge_code"] = df["gauge_code"].astype(str)
        df.drop_duplicates(subset=["gauge_code"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Write cache for future executions if unfiltered
        if use_cache and states is None and years is None and not df.empty:
            try:
                df.to_parquet(cache_file, index=False)
            except Exception:
                pass

        return df

    # Fallback to legacy HDF5
    try:
        df = pd.read_hdf(HDF5_PATH, "table_info")
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(
            f"Failed to load gauge info from UNIPLU or HDF5 ({HDF5_PATH}): {e}"
        )
    df["gauge_code"] = df["gauge_code"].astype(str)
    if states is not None:
        df = df[df["state"].isin(states)]
    return df


# ─── Public Loaders: Rainfall Time Series Data ────────────────────────

def load_rainfall_data(
    gauge_codes: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
    chunksize: int = 0,
) -> pd.DataFrame:
    """Load rainfall timeseries data with support for direct UNIPLU reading.

    Parameters
    ----------
    gauge_codes : list of str, optional
        List of gauge codes to retrieve.
    states : list of str, optional
        Filter by Brazilian states (e.g. ['SP', 'MG']).
    years : list of int, optional
        Filter by years (e.g. [2023, 2024]).
    start_date : str, optional
        Start datetime filter (inclusive).
    end_date : str, optional
        End datetime filter (inclusive).
    sample : int, optional
        Number of stations to sample randomly.
    seed : int, optional
        Random seed for sampling.
    chunksize : int, optional
        HDF5 chunksize legacy option.
    """
    if _is_uniplu_available():
        target_codes = set(str(c) for c in gauge_codes) if gauge_codes else None

        # Sample stations if requested
        if sample is not None:
            df_info = load_gauge_info(states=states, years=years)
            all_codes = df_info["gauge_code"].unique()
            if target_codes:
                all_codes = np.array([c for c in all_codes if c in target_codes])
            rng = np.random.default_rng(seed)
            if len(all_codes) > 0:
                sampled_codes = rng.choice(all_codes, min(sample, len(all_codes)), replace=False)
                target_codes = set(sampled_codes)
            else:
                return pd.DataFrame(columns=["gauge_code", "datetime", "rain_mm"])

        # If gauge_codes specified without states, identify states from metadata to avoid scanning all zips
        if target_codes and not states:
            df_info = load_gauge_info(years=years)
            matched = df_info[df_info["gauge_code"].isin(target_codes)]
            if not matched.empty:
                states = list(matched["state"].dropna().unique())

        # If dates given without years, infer years
        if years is None:
            inferred_years = set()
            if start_date:
                inferred_years.add(pd.to_datetime(start_date).year)
            if end_date:
                inferred_years.add(pd.to_datetime(end_date).year)
            if len(inferred_years) == 1:
                years = list(inferred_years)
            elif len(inferred_years) > 1:
                y_min, y_max = min(inferred_years), max(inferred_years)
                years = list(range(y_min, y_max + 1))

        zip_paths = _filter_uniplu_zips(states=states, years=years)
        if not zip_paths:
            return pd.DataFrame(columns=["gauge_code", "datetime", "rain_mm"])

        pieces = []
        codes_list = list(target_codes) if target_codes else None

        for zp in zip_paths:
            try:
                with zipfile.ZipFile(zp) as zf:
                    if "table_data.parquet" not in zf.namelist():
                        continue
                    stream = io.BytesIO(zf.read("table_data.parquet"))
                    
                    filters = None
                    if codes_list:
                        filters = [("gauge_code", "in", codes_list)]

                    table = pq.read_table(
                        stream,
                        columns=["gauge_code", "datetime", "rain_mm"],
                        filters=filters,
                    )
                    if table.num_rows > 0:
                        df_piece = table.to_pandas()
                        pieces.append(df_piece)
            except Exception:
                continue

        if not pieces:
            return pd.DataFrame(columns=["gauge_code", "datetime", "rain_mm"])

        df = pd.concat(pieces, ignore_index=True)
        df["gauge_code"] = df["gauge_code"].astype(str)
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Temporal post-filters
        if start_date is not None:
            df = df[df["datetime"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df["datetime"] <= pd.to_datetime(end_date)]

        return df

    # Legacy HDF5 loading
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


# ─── Public Loaders: Daily and Aggregated Data ────────────────────────

def load_daily_data(
    gauge_codes: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Load daily aggregated rainfall data.

    If reading from UNIPLU, aggregates sub-daily timeseries to daily sums.
    Otherwise reads from legacy HDF5 or parquet.
    """
    if _is_uniplu_available():
        df_rainfall = load_rainfall_data(
            gauge_codes=gauge_codes,
            states=states,
            years=years,
        )
        if df_rainfall.empty:
            return pd.DataFrame(columns=["gauge_code", "date", "rain_mm"])

        df_rainfall["date"] = df_rainfall["datetime"].dt.floor("D")
        daily = (
            df_rainfall.groupby(["gauge_code", "date"], as_index=False)["rain_mm"]
            .sum()
        )
        return daily

    try:
        df = pd.read_hdf(HDF5_PATH, "table_data_daily")
    except (FileNotFoundError, KeyError) as e:
        raise FileNotFoundError(f"Failed to load daily data from {HDF5_PATH}: {e}")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    if gauge_codes is not None:
        df = df[df["gauge_code"].isin(gauge_codes)]
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


# ─── Utility Query Functions ─────────────────────────────────────────

def get_gauge_data_by_code(df_rainfall: pd.DataFrame, code: str) -> pd.DataFrame:
    filt = df_rainfall["gauge_code"] == str(code)
    temp = df_rainfall[filt].copy()
    temp.sort_values("datetime", ascending=True, inplace=True)
    temp.reset_index(drop=True, inplace=True)
    return temp


def get_stations_by_network(gauge_info: pd.DataFrame, network: str) -> List[str]:
    filt = gauge_info["network"] == network
    return list(gauge_info[filt]["gauge_code"].unique())


# ─── Legacy Parquet directory loaders ────────────────────────────────

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

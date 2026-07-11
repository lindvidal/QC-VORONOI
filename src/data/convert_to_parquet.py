"""
Convert UNIPLU HDF5 to partitioned Parquet.

Usage:
    python -m src.data.convert_to_parquet             # full conversion
    python -m src.data.convert_to_parquet --sample 5  # just 5 states for testing

Output structure:
    src/data/parquet/
    ├── table_info.parquet
    ├── table_data_daily.parquet
    ├── table_data/
    │   ├── state=AC/
    │   │   └── year=2023.parquet
    │   ├── state=AL/...
    │   └── ...
    ├── table_data_daily_hq.parquet
    └── table_data_monthly_hq.parquet
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from ..config import HDF5_PATH, PARQUET_DIR


def convert_table_info():
    print("[1/5] Converting table_info...")
    df = pd.read_hdf(HDF5_PATH, "table_info")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df.to_parquet(PARQUET_DIR / "table_info.parquet", index=False)
    print(f"       {len(df)} rows → table_info.parquet")


def convert_table_data(sample_states: int = 0):
    print("[2/5] Converting table_data (partitioned by state/year)...")
    info = pd.read_hdf(HDF5_PATH, "table_info")[["gauge_code", "state"]]
    info["gauge_code"] = info["gauge_code"].astype(str)

    states = sorted(info["state"].unique())
    if sample_states > 0:
        states = states[:sample_states]
        print(f"       Sample mode: {sample_states} states")

    total = 0
    for state in states:
        codes = info[info["state"] == state]["gauge_code"].tolist()
        if not codes:
            continue
        where = "(" + " | ".join([f"gauge_code == '{c}'" for c in codes]) + ")"
        t0 = time.time()
        state_df = pd.read_hdf(HDF5_PATH, "table_data", where=where)
        state_df["gauge_code"] = state_df["gauge_code"].astype(str)
        state_df["datetime"] = pd.to_datetime(state_df["datetime"])
        state_df["year"] = state_df["datetime"].dt.year

        for year in sorted(state_df["year"].unique()):
            subset = state_df[state_df["year"] == year][["gauge_code", "datetime", "rain_mm"]]
            out_dir = PARQUET_DIR / "table_data" / f"state={state}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"year={year}.parquet"
            subset.to_parquet(out_path, index=False)
        elapsed = time.time() - t0
        total += len(state_df)
        print(f"       {state}: {len(state_df):>8,} rows  ({elapsed:.1f}s)")

    if total:
        print(f"       Total: {total:>8,} rows")


def convert_table_data_daily():
    print("[3/5] Converting table_data_daily...")
    df = pd.read_hdf(HDF5_PATH, "table_data_daily")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(PARQUET_DIR / "table_data_daily.parquet", index=False)
    print(f"       {len(df)} rows → table_data_daily.parquet")


def convert_table_data_daily_hq():
    print("[4/5] Converting table_data_daily_hq...")
    df = pd.read_hdf(HDF5_PATH, "table_data_daily_hq")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(PARQUET_DIR / "table_data_daily_hq.parquet", index=False)
    print(f"       {len(df)} rows → table_data_daily_hq.parquet")


def convert_table_data_monthly_hq():
    print("[5/5] Converting table_data_monthly_hq...")
    df = pd.read_hdf(HDF5_PATH, "table_data_monthly_hq")
    df["gauge_code"] = df["gauge_code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(PARQUET_DIR / "table_data_monthly_hq.parquet", index=False)
    print(f"       {len(df)} rows → table_data_monthly_hq.parquet")


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 to partitioned Parquet")
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of states to process (0 = all)")
    args = parser.parse_args()

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Converting {HDF5_PATH}")
    print(f"Output: {PARQUET_DIR}\n")

    t_start = time.time()

    convert_table_info()
    convert_table_data(sample_states=args.sample)
    convert_table_data_daily()
    convert_table_data_daily_hq()
    convert_table_data_monthly_hq()

    total = time.time() - t_start
    print(f"\nDone in {total:.1f}s")
    print(f"Output: {PARQUET_DIR}")


if __name__ == "__main__":
    main()

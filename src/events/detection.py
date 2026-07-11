import pandas as pd
import numpy as np
import calendar
from typing import List

from ..config import MINIMUM_DEPTH_MM


def _create_rainfall_event(
    gauge_code: str,
    date_hour_begin: pd.Timestamp,
    date_hour_end: pd.Timestamp,
    rain: float,
    rainfall_registers: int,
    rainfall_time_step: pd.Timedelta,
) -> pd.DataFrame:
    date_hour_end_adj = date_hour_end + rainfall_time_step
    dif = date_hour_end_adj - date_hour_begin
    duration_hours = dif.total_seconds() / 3600.0
    intensity = rain / duration_hours if duration_hours > 0 else 0.0
    effective_duration = rainfall_time_step.seconds / 3600.0 * rainfall_registers
    dry_time = duration_hours - effective_duration
    record = {
        "gauge_code": gauge_code,
        "date_hour_begin": date_hour_begin,
        "rain_mm": rain,
        "duration_hour": duration_hours,
        "intensity_mm_hour": intensity,
        "dry_time_hour": dry_time,
        "rainfall_of_pulses": rainfall_registers,
    }
    return pd.DataFrame([record])


def determine_rainfall_events(
    mit: pd.Timedelta,
    df_rainfall_data: pd.DataFrame,
    rainfall_time_step: pd.Timedelta,
) -> pd.DataFrame:
    if len(df_rainfall_data) < 2:
        return pd.DataFrame(
            columns=[
                "gauge_code",
                "date_hour_begin",
                "rain_mm",
                "duration_hour",
                "intensity_mm_hour",
                "dry_time_hour",
                "rainfall_of_pulses",
            ]
        )
    events = []
    code = df_rainfall_data.iloc[0]["gauge_code"]
    begin = begin_a = df_rainfall_data.iloc[0]["datetime"]
    rain_count = 0
    tot_rain = 0.0
    n = len(df_rainfall_data) - 1
    for index in range(n):
        row = df_rainfall_data.iloc[index]
        rain = row["rain_mm"]
        tot_rain += rain
        end_a = begin_a + rainfall_time_step
        begin_b = df_rainfall_data.iloc[index + 1]["datetime"]
        dif = begin_b - end_a
        if dif <= mit:
            rain_count += 1
            begin_a = begin_b
        else:
            begin_a = df_rainfall_data.iloc[index]["datetime"]
            ev = _create_rainfall_event(
                code, begin, begin_a, tot_rain, rain_count + 1, rainfall_time_step
            )
            events.append(ev)
            begin = begin_a = df_rainfall_data.iloc[index + 1]["datetime"]
            rain_count = 0
            tot_rain = 0.0
    tot_rain += df_rainfall_data.iloc[n]["rain_mm"]
    ev = _create_rainfall_event(
        code, begin, begin_a, tot_rain, rain_count + 1, rainfall_time_step
    )
    events.append(ev)
    if events:
        return pd.concat(events, ignore_index=True)
    return pd.DataFrame(
        columns=[
            "gauge_code",
            "date_hour_begin",
            "rain_mm",
            "duration_hour",
            "intensity_mm_hour",
            "dry_time_hour",
            "rainfall_of_pulses",
        ]
    )


def compute_event_statistics(
    df_events: pd.DataFrame, gauge_info: pd.Series, mit_minutes: int
) -> dict:
    df = df_events[df_events["rain_mm"] >= MINIMUM_DEPTH_MM].copy()
    if df.empty:
        return None
    stats = df[["rain_mm", "duration_hour", "intensity_mm_hour", "dry_time_hour"]].describe()
    result = {
        "gauge_code": gauge_info["gauge_code"],
        "city": gauge_info["city"],
        "state": gauge_info["state"],
        "latitude": gauge_info["lat"],
        "longitude": gauge_info["long"],
        "elevation": gauge_info["elevation"],
        "network": gauge_info["network"],
        "mit_minutes": mit_minutes,
        "yearly_rainfall": df["rain_mm"].sum(),
        "rainfall_events": df.shape[0],
        "mean_rainfall_depth": stats.loc["mean", "rain_mm"],
        "mean_rainfall_duration": stats.loc["mean", "duration_hour"],
        "mean_rainfall_intensity": stats.loc["mean", "intensity_mm_hour"],
        "mean_dry_time": stats.loc["mean", "dry_time_hour"],
        "sd_rainfall_depth": stats.loc["std", "rain_mm"],
        "sd_rainfall_duration": stats.loc["std", "duration_hour"],
        "sd_rainfall_intensity": stats.loc["std", "intensity_mm_hour"],
        "sd_rainfall_dry_time": stats.loc["std", "dry_time_hour"],
        "max_rainfall_depth": stats.loc["max", "rain_mm"],
        "max_rainfall_duration": stats.loc["max", "duration_hour"],
        "max_rainfall_intensity": stats.loc["max", "intensity_mm_hour"],
        "max_dry_time": stats.loc["max", "dry_time_hour"],
    }
    return result


def count_valid_days(df_rainfall: pd.DataFrame) -> int:
    return len(df_rainfall["datetime"].dt.date.unique())


def create_days_record(year: int, code: str, n_days: int) -> pd.DataFrame:
    missing = (366 if calendar.isleap(year) else 365) - n_days
    return pd.DataFrame({"year": [year], "code": [code], "ndays": [n_days], "missing_days": [missing]})


def compute_rainfall_time_step(df_rainfall: pd.DataFrame) -> pd.Timedelta:
    if len(df_rainfall) < 2:
        return pd.Timedelta(15, unit="m")
    return df_rainfall["datetime"].iloc[1] - df_rainfall["datetime"].iloc[0]


def process_station_events(
    df_rainfall: pd.DataFrame,
    mit_minutes: int,
    gauge_info_row: pd.Series,
) -> dict:
    rainfall_data = df_rainfall[df_rainfall["rain_mm"] > 0].copy()
    rainfall_data.reset_index(drop=True, inplace=True)
    if len(rainfall_data) < 2:
        return None
    time_step = compute_rainfall_time_step(df_rainfall)
    mit = pd.Timedelta(mit_minutes, unit="m")
    events = determine_rainfall_events(mit, rainfall_data, time_step)
    if events.empty:
        return None
    return compute_event_statistics(events, gauge_info_row, mit_minutes)

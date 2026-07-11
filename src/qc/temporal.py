import pandas as pd

from ..config import MIN_VALID_DAYS, MAX_RAIN_EVENT_MM, ANNUAL_RAIN_MIN, ANNUAL_RAIN_MAX, MIN_DAY_COUNT


def check_valid_days(n_days: int) -> bool:
    return n_days >= MIN_VALID_DAYS


def prefilter_station(
    yearly_rainfall: float,
    max_rainfall_depth: float,
    day_count: int,
) -> bool:
    if yearly_rainfall < ANNUAL_RAIN_MIN or yearly_rainfall > ANNUAL_RAIN_MAX:
        return False
    if max_rainfall_depth > MAX_RAIN_EVENT_MM:
        return False
    if day_count < MIN_DAY_COUNT:
        return False
    return True


def temporal_consistency_check(df_daily: pd.DataFrame, gauge_code: str) -> dict:
    data = df_daily[df_daily["gauge_code"] == gauge_code].copy()
    if data.empty:
        return {"gauge_code": gauge_code, "valid": False, "reason": "no_data"}
    data = data.sort_values("date")
    data["rain_mm"] = pd.to_numeric(data["rain_mm"], errors="coerce")
    total_rain = data["rain_mm"].sum()
    n_days = data["rain_mm"].notna().sum()
    n_rainy = (data["rain_mm"] > 0).sum()
    max_daily = data["rain_mm"].max()
    gaps = data["date"].diff().dt.days
    max_gap = gaps.max() if len(gaps) > 1 else 0
    return {
        "gauge_code": gauge_code,
        "total_days": len(data),
        "valid_days": n_days,
        "rainy_days": n_rainy,
        "total_rainfall_mm": total_rain,
        "max_daily_rainfall_mm": max_daily,
        "max_gap_days": max_gap,
        "valid": n_days >= MIN_DAY_COUNT and total_rain >= ANNUAL_RAIN_MIN,
    }

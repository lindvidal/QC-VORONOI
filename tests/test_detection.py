import unittest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from src.events.detection import (
    determine_rainfall_events,
    compute_event_statistics,
    count_valid_days,
    create_days_record,
    compute_rainfall_time_step,
)
from src.config import MINIMUM_DEPTH_MM


class TestDetermineRainfallEvents(unittest.TestCase):

    def setUp(self):
        base = pd.Timestamp("2023-01-01")
        self.df_15min = pd.DataFrame({
            "gauge_code": ["A"] * 10,
            "datetime": [base + pd.Timedelta(minutes=15 * i) for i in range(10)],
            "rain_mm": [0.0, 2.0, 3.0, 0.0, 0.0, 5.0, 4.0, 0.0, 0.0, 1.0],
        })
        self.time_step = pd.Timedelta(15, unit="m")

    def test_empty_data(self):
        df_empty = pd.DataFrame(columns=["gauge_code", "datetime", "rain_mm"])
        result = determine_rainfall_events(pd.Timedelta(60, unit="m"), df_empty, self.time_step)
        self.assertTrue(result.empty)

    def test_single_row(self):
        df_one = self.df_15min.iloc[:1]
        result = determine_rainfall_events(pd.Timedelta(60, unit="m"), df_one, self.time_step)
        self.assertTrue(result.empty)

    def test_detects_multiple_events(self):
        mit = pd.Timedelta(60, unit="m")
        events = determine_rainfall_events(mit, self.df_15min, self.time_step)
        self.assertGreater(len(events), 0)
        self.assertIn("gauge_code", events.columns)
        self.assertIn("rain_mm", events.columns)

    def test_event_properties(self):
        mit = pd.Timedelta(120, unit="m")
        events = determine_rainfall_events(mit, self.df_15min, self.time_step)
        for _, ev in events.iterrows():
            self.assertGreater(ev["rain_mm"], 0)
            self.assertGreater(ev["duration_hour"], 0)
            self.assertGreaterEqual(ev["intensity_mm_hour"], 0)


class TestComputeEventStatistics(unittest.TestCase):

    def test_returns_none_for_empty(self):
        df_events = pd.DataFrame(columns=["rain_mm", "duration_hour", "intensity_mm_hour", "dry_time_hour"])
        gauge_info = pd.Series({"gauge_code": "A", "city": "X", "state": "Y",
                                "lat": -10.0, "long": -50.0, "elevation": 100, "network": "Test"})
        result = compute_event_statistics(df_events, gauge_info, 30)
        self.assertIsNone(result)

    def test_valid_statistics(self):
        events = pd.DataFrame({
            "rain_mm": [10.0, 20.0],
            "duration_hour": [1.0, 2.0],
            "intensity_mm_hour": [10.0, 10.0],
            "dry_time_hour": [5.0, 10.0],
        })
        gauge_info = pd.Series({"gauge_code": "A", "city": "X", "state": "Y",
                                "lat": -10.0, "long": -50.0, "elevation": 100, "network": "Test"})
        result = compute_event_statistics(events, gauge_info, 30)
        self.assertIsNotNone(result)
        self.assertEqual(result["gauge_code"], "A")
        self.assertAlmostEqual(result["yearly_rainfall"], 30.0)
        self.assertEqual(result["rainfall_events"], 2)
        self.assertAlmostEqual(result["mean_rainfall_depth"], 15.0)


class TestCountValidDays(unittest.TestCase):

    def test_count_valid_days(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=5, freq="D"),
        })
        self.assertEqual(count_valid_days(df), 5)

    def test_count_valid_days_with_duplicates(self):
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
            ],
        })
        self.assertEqual(count_valid_days(df), 2)


class TestCreateDaysRecord(unittest.TestCase):

    def test_leap_year(self):
        record = create_days_record(2024, "A", 300)
        self.assertEqual(record["year"].iloc[0], 2024)
        self.assertEqual(record["code"].iloc[0], "A")
        self.assertEqual(record["ndays"].iloc[0], 300)
        self.assertEqual(record["missing_days"].iloc[0], 66)

    def test_non_leap_year(self):
        record = create_days_record(2023, "A", 300)
        self.assertEqual(record["missing_days"].iloc[0], 65)


class TestComputeRainfallTimeStep(unittest.TestCase):

    def test_regular_step(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=3, freq="15min"),
            "rain_mm": [0.0, 1.0, 2.0],
        })
        step = compute_rainfall_time_step(df)
        self.assertEqual(step, pd.Timedelta(15, unit="m"))

    def test_single_row(self):
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2023-01-01")],
            "rain_mm": [0.0],
        })
        step = compute_rainfall_time_step(df)
        self.assertEqual(step, pd.Timedelta(15, unit="m"))


if __name__ == "__main__":
    unittest.main()

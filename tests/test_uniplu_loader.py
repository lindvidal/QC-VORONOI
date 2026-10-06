import unittest
import pandas as pd
from pathlib import Path

from src.data.loader import (
    load_gauge_info,
    load_rainfall_data,
    load_daily_data,
    get_gauge_data_by_code,
)
from src.config import UNIPLU_DIR


class TestUNIPLULoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uniplu_exists = UNIPLU_DIR.exists() and any(UNIPLU_DIR.glob("*.zip"))

    def test_load_gauge_info(self):
        if not self.uniplu_exists:
            self.skipTest("UNIPLU directory not found")

        df_info = load_gauge_info(states=["AC"], years=[2025])
        self.assertIsInstance(df_info, pd.DataFrame)
        self.assertFalse(df_info.empty)
        self.assertIn("gauge_code", df_info.columns)
        self.assertIn("lat", df_info.columns)
        self.assertIn("long", df_info.columns)
        self.assertTrue((df_info["state"] == "AC").all())

    def test_load_rainfall_data_with_sample(self):
        if not self.uniplu_exists:
            self.skipTest("UNIPLU directory not found")

        sample_size = 2
        df_rain = load_rainfall_data(
            states=["AC"],
            years=[2025],
            sample=sample_size,
            seed=42,
        )
        self.assertIsInstance(df_rain, pd.DataFrame)
        self.assertFalse(df_rain.empty)
        self.assertIn("gauge_code", df_rain.columns)
        self.assertIn("datetime", df_rain.columns)
        self.assertIn("rain_mm", df_rain.columns)
        self.assertEqual(df_rain["gauge_code"].nunique(), sample_size)

    def test_load_daily_data(self):
        if not self.uniplu_exists:
            self.skipTest("UNIPLU directory not found")

        df_daily = load_daily_data(states=["AC"], years=[2025])
        self.assertIsInstance(df_daily, pd.DataFrame)
        self.assertFalse(df_daily.empty)
        self.assertIn("gauge_code", df_daily.columns)
        self.assertIn("date", df_daily.columns)
        self.assertIn("rain_mm", df_daily.columns)


if __name__ == "__main__":
    unittest.main()

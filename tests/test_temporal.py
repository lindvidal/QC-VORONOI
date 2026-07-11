import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from src.qc.temporal import check_valid_days, prefilter_station


class TestCheckValidDays(unittest.TestCase):

    def test_valid_days_above_threshold(self):
        self.assertTrue(check_valid_days(310))

    def test_valid_days_at_threshold(self):
        self.assertTrue(check_valid_days(305))

    def test_valid_days_below_threshold(self):
        self.assertFalse(check_valid_days(200))

    def test_valid_days_zero(self):
        self.assertFalse(check_valid_days(0))


class TestPrefilterStation(unittest.TestCase):

    def test_passes_valid_station(self):
        self.assertTrue(prefilter_station(1000, 30, 350))

    def test_fails_low_rainfall(self):
        self.assertFalse(prefilter_station(100, 30, 350))

    def test_fails_high_rainfall(self):
        self.assertFalse(prefilter_station(5000, 30, 350))

    def test_fails_high_event(self):
        self.assertFalse(prefilter_station(1000, 100, 350))

    def test_fails_low_day_count(self):
        self.assertFalse(prefilter_station(1000, 30, 100))

    def test_boundary_min_rainfall(self):
        self.assertTrue(prefilter_station(300, 40, 300))

    def test_boundary_max_rainfall(self):
        self.assertTrue(prefilter_station(3000, 40, 300))


if __name__ == "__main__":
    unittest.main()

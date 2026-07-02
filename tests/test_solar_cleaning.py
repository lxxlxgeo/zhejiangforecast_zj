from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from zhejiangforecast_zj.algorithm_engine.adapters.cleaning import clean_solar_power


class SolarCleaningTest(unittest.TestCase):
    def test_clean_solar_power_uses_irradiance_rules(self) -> None:
        times = pd.date_range("2026-06-01", periods=192, freq="15min")
        hour = times.hour + times.minute / 60.0
        daylight = (hour >= 6.0) & (hour <= 18.0)
        irradiance = np.where(daylight, np.sin((hour - 6.0) / 12.0 * np.pi) * 900.0, 0.0)
        power = irradiance / 900.0 * 80.0
        frame = pd.DataFrame(
            {
                "time_bj": times,
                "time_utc": times - pd.Timedelta(hours=8),
                "power_mw": power,
                "direct_irradiance": irradiance,
            }
        )
        frame.loc[40, "power_mw"] = -5.0
        frame.loc[80, "power_mw"] = 120.0
        frame.loc[120, "direct_irradiance"] = np.nan

        with tempfile.TemporaryDirectory() as tmp:
            result = clean_solar_power(frame, Path(tmp), capacity_mw=100.0)

        self.assertTrue(result.used_external_pipeline)
        self.assertEqual(result.summary["capacity_mw"], 100.0)
        self.assertLess(result.summary["clean_rows"], len(frame))
        self.assertGreater(result.summary["clean_rows"], 100)
        self.assertIn("direct_irradiance", result.clean_power.columns)
        self.assertGreater(result.summary["flag_counts"]["flag_power_negative"], 0)
        self.assertGreater(result.summary["flag_counts"]["flag_power_over_capacity"], 0)
        self.assertGreater(result.summary["flag_counts"]["flag_missing"], 0)


if __name__ == "__main__":
    unittest.main()

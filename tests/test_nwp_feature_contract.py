from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from zhejiangforecast_zj.algorithm_engine.adapters.nwp import _align_tensor_channels, assert_nwp_only_features, nwp_ml_feature_names
from zhejiangforecast_zj.services.inference import _is_nwp_feature_set


class NwpFeatureContractTest(unittest.TestCase):
    def test_nwp_ml_feature_names_exclude_power_history_and_metadata(self) -> None:
        frame = pd.DataFrame(
            {
                "time_bj": ["2025-10-21 00:00:00"],
                "time_utc": ["2025-10-20 16:00:00"],
                "power_mw": [1.0],
                "issue_time_utc": ["2025-10-20T12:00:00+00:00"],
                "horizon_code": ["N1"],
                "nwp_file": ["2025102012.nc"],
                "lead_hours": [4.0],
                "nwp_ssrd_mean": [10.0],
                "nwp_t2m_center_t0": [280.0],
                "history_power_lag_1": [1.0],
                "history_power_roll96": [2.0],
                "capacity_mw": [100.0],
            }
        )

        self.assertEqual(nwp_ml_feature_names(frame), ["lead_hours", "nwp_ssrd_mean", "nwp_t2m_center_t0"])

    def test_nwp_feature_contract_rejects_non_nwp_fields(self) -> None:
        assert_nwp_only_features(["lead_hours", "nwp_ssrd_mean"])
        self.assertTrue(_is_nwp_feature_set(["lead_hours", "nwp_ssrd_mean"]))
        self.assertFalse(_is_nwp_feature_set(["lead_hours", "history_power_lag_1"]))
        with self.assertRaises(ValueError):
            assert_nwp_only_features(["nwp_ssrd_mean", "history_power_lag_1"])

    def test_align_tensor_channels_selects_first_sample_schema(self) -> None:
        tensor = np.arange(4 * 2 * 3 * 3).reshape(4, 2, 3, 3)

        aligned, names = _align_tensor_channels(tensor, ["ssrd", "fdir", "t2m", "sp"], ["ssrd", "t2m"])

        self.assertEqual(names, ["ssrd", "t2m"])
        np.testing.assert_array_equal(aligned[0], tensor[0])
        np.testing.assert_array_equal(aligned[1], tensor[2])

    def test_align_tensor_channels_rejects_missing_required_channel(self) -> None:
        tensor = np.zeros((2, 2, 3, 3))

        with self.assertRaisesRegex(ValueError, "missing channels"):
            _align_tensor_channels(tensor, ["ssrd", "t2m"], ["ssrd", "sp"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from zhejiangforecast_zj.core.time_semantics import (
    horizon_for_valid_time,
    issue_for_valid_day_bj,
    valid_window_for_issue,
)


class TimeSemanticsTest(unittest.TestCase):
    def test_n1_window_from_12z_issue(self) -> None:
        issue = datetime(2026, 6, 28, 12, tzinfo=timezone.utc)
        window = valid_window_for_issue(issue, "N1")
        self.assertEqual(str(window.start_bj), "2026-06-29 00:00:00+08:00")
        self.assertEqual(str(window.end_bj), "2026-06-29 23:45:00+08:00")
        self.assertEqual(window.lead_start_hours, 4.0)
        self.assertEqual(window.lead_end_hours, 27.75)

    def test_issue_for_valid_bj_day(self) -> None:
        issue = issue_for_valid_day_bj(date(2026, 6, 29), "N1")
        self.assertEqual(issue, datetime(2026, 6, 28, 12, tzinfo=timezone.utc))

    def test_horizon_for_valid_time(self) -> None:
        issue = datetime(2026, 6, 28, 12, tzinfo=timezone.utc)
        valid = datetime(2026, 6, 29, 3, tzinfo=timezone.utc)
        self.assertEqual(horizon_for_valid_time(issue, valid), "N1")


if __name__ == "__main__":
    unittest.main()


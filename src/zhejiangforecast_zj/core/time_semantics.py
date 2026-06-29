from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


BJ_TZ = ZoneInfo("Asia/Shanghai")
UTC = timezone.utc
HORIZON_DAY_INDEX = {"N1": 1, "N2": 2, "N3": 3}


@dataclass(frozen=True)
class HorizonWindow:
    horizon_code: str
    issue_time_utc: datetime
    start_bj: datetime
    end_bj: datetime
    start_utc: datetime
    end_utc: datetime
    lead_start_hours: float
    lead_end_hours: float


def require_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def issue_for_valid_day_bj(valid_day_bj: date, horizon_code: str = "N1", cycle_hour: int = 12) -> datetime:
    """Return the ECMWF UTC cycle used for a Beijing valid calendar day.

    In the current business convention, N1 means the next Beijing calendar day
    covered by the 12Z HRES cycle. Example: issue 2026-06-28 12Z maps to N1
    valid window 2026-06-29 00:00-23:45 Beijing time.
    """

    code = horizon_code.upper()
    if code not in HORIZON_DAY_INDEX:
        raise ValueError(f"Unsupported horizon code: {horizon_code}")
    issue_day = valid_day_bj - timedelta(days=HORIZON_DAY_INDEX[code])
    return datetime.combine(issue_day, time(hour=cycle_hour), tzinfo=UTC)


def valid_window_for_issue(issue_time_utc: datetime, horizon_code: str = "N1") -> HorizonWindow:
    code = horizon_code.upper()
    if code not in HORIZON_DAY_INDEX:
        raise ValueError(f"Unsupported horizon code: {horizon_code}")
    issue_utc = require_aware_utc(issue_time_utc)
    issue_bj_date = issue_utc.astimezone(BJ_TZ).date()
    valid_day_bj = issue_bj_date + timedelta(days=HORIZON_DAY_INDEX[code])
    start_bj = datetime.combine(valid_day_bj, time(0, 0), tzinfo=BJ_TZ)
    end_bj = start_bj + timedelta(hours=23, minutes=45)
    start_utc = start_bj.astimezone(UTC)
    end_utc = end_bj.astimezone(UTC)
    return HorizonWindow(
        horizon_code=code,
        issue_time_utc=issue_utc,
        start_bj=start_bj,
        end_bj=end_bj,
        start_utc=start_utc,
        end_utc=end_utc,
        lead_start_hours=(start_utc - issue_utc).total_seconds() / 3600.0,
        lead_end_hours=(end_utc - issue_utc).total_seconds() / 3600.0,
    )


def horizon_for_valid_time(issue_time_utc: datetime, valid_time_utc: datetime) -> str | None:
    valid_bj_day = require_aware_utc(valid_time_utc).astimezone(BJ_TZ).date()
    issue_bj_day = require_aware_utc(issue_time_utc).astimezone(BJ_TZ).date()
    delta = (valid_bj_day - issue_bj_day).days
    for code, days in HORIZON_DAY_INDEX.items():
        if days == delta:
            return code
    return None


def lead_hours(issue_time_utc: datetime, valid_time_utc: datetime) -> float:
    return (require_aware_utc(valid_time_utc) - require_aware_utc(issue_time_utc)).total_seconds() / 3600.0


from __future__ import annotations

import re


def parse_coordinate(value: object) -> float | None:
    """Parse decimal degrees or DMS strings such as 120°36'08.736477"."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    pattern = re.compile(
        r"(?P<deg>-?\d+(?:\.\d+)?)\s*[°度]?\s*"
        r"(?:(?P<min>\d+(?:\.\d+)?)\s*['′分])?\s*"
        r"(?:(?P<sec>\d+(?:\.\d+)?)\s*(?:[\"″秒])?)?"
    )
    match = pattern.search(text)
    if not match:
        return None
    deg = float(match.group("deg"))
    minutes = float(match.group("min") or 0.0)
    seconds = float(match.group("sec") or 0.0)
    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minutes / 60.0 + seconds / 3600.0)


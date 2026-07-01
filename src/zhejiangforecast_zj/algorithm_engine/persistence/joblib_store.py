from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def dump_joblib(payload: Any, path: str | Path, compress: int | tuple[str, int] = 3) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, target, compress=compress)
    return target


def load_joblib(path: str | Path) -> Any:
    return joblib.load(Path(path))


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass
class ExperimentRecord:
    name: str
    config_path: str
    input_path: str
    output_path: str
    metrics_path: str | None = None
    notes: str = ""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

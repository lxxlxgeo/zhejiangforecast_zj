from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import SolarCleanConfig
from .methods import (
    add_basic_flags,
    add_curve_flags,
    add_ramp_flags,
    add_repeat_flags,
    finalize_flags,
    normalize_frame,
    summarize_flags,
)


@dataclass
class SolarCleanResult:
    cleaned: pd.DataFrame
    summary: dict
    output_dir: Path


def clean_dataframe(frame: pd.DataFrame, cfg: SolarCleanConfig) -> SolarCleanResult:
    work = normalize_frame(frame, cfg)
    work = add_basic_flags(work, cfg)
    work = add_curve_flags(work, cfg)
    work = add_repeat_flags(work, cfg)
    work = add_ramp_flags(work, cfg)
    work = finalize_flags(work)
    return SolarCleanResult(cleaned=work, summary=summarize_flags(work, cfg), output_dir=Path("."))


def run_pipeline(input_path: str | Path, out_dir: str | Path, cfg: SolarCleanConfig) -> SolarCleanResult:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(input_path)
    result = clean_dataframe(frame, cfg)
    result.output_dir = out_dir
    result.cleaned.to_csv(out_dir / "cleaned_15min.csv", index=False, encoding="utf-8-sig")
    (out_dir / "solar_clean_summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


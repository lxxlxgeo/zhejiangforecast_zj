from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from power_ml_baseline.evaluation.metrics import compute_metrics, segment_metrics_by_power


def write_predictions(path: str | Path, times, y_true, y_pred) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"time": pd.to_datetime(times), "y_true": y_true, "y_pred": y_pred, "error": np.asarray(y_pred) - np.asarray(y_true)})
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_metrics_report(out_dir: str | Path, y_true, y_pred, capacity: float | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, capacity)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    seg = segment_metrics_by_power(y_true, y_pred, capacity)
    if not seg.empty:
        seg.to_csv(out_dir / "metrics_by_power_segment.csv", index=False, encoding="utf-8-sig")
    return metrics

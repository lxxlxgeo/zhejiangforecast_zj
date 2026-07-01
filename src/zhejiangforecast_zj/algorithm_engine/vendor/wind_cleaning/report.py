from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd


def _table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + df.to_csv(index=False).strip() + "\n```"


def write_report(out_dir: str | Path, summary: Dict[str, Any], method_cmp: pd.DataFrame, keep_rate: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    lines = []
    lines.append("# 风电场平均风速-功率混合自动清洗报告\n")
    lines.append("## 1. 运行摘要\n")
    lines.append(f"- 装机容量：{summary.get('capacity_mw')} MW")
    lines.append(f"- 物理有效样本：{summary.get('valid_rows'):,}")
    lines.append(f"- 最终保留样本：{summary.get('clean_rows'):,}")
    lines.append(f"- 最终剔除样本：{summary.get('removed_rows'):,}")
    lines.append(f"- 剔除率：{summary.get('removed_rate_pct'):.2f}%")
    lines.append(f"- 自动识别低风速/切入过渡区终点：{summary.get('transition_end_ws')} m/s")
    lines.append(f"- 自动识别接近额定区起点：{summary.get('rated_start_ws')} m/s")
    lines.append("\n## 2. 方法对比\n")
    lines.append(_table_text(method_cmp))
    lines.append("\n## 3. 分风速区间保留率\n")
    lines.append(_table_text(keep_rate))
    lines.append("\n## 4. 输出说明\n")
    lines.append("- `cleaned_15min.csv`：清洗结果主表。")
    lines.append("- `curve_grid.csv`：q50/q90 单调功率曲线。")
    lines.append("- `method_comparison.csv`：单方法和融合方法的标记数量。")
    lines.append("- `keep_rates_by_ws.csv`：按风速区间统计保留率。")
    lines.append("- `main_cleaning.png`、`lowwind_zoom.png`、`removed_points.png`：可视化结果。")
    lines.append("\n## 5. 使用建议\n")
    lines.append("本方案适合在只拿到场站平均风速和场站实发功率时做自动化清洗。低风速段提高删除门槛，中高风速段通过多模型投票清理低功率水平带、稀疏异常和局部密度异常。")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

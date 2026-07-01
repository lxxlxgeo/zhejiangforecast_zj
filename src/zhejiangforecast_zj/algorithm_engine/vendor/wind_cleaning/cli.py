from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="风电场平均风速-功率混合自动清洗工程")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="执行清洗流程")
    run.add_argument("--qc", required=True, help="场站实际功率 CSV，例如 res_qc_15min.csv")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--mean-ws", help="已聚合好的场站平均风速 CSV，至少包含 data_time, ws_mean")
    source.add_argument("--fan", help="单风机风速 CSV，至少包含 data_time, fan_no, wind_speed")
    run.add_argument("--out", required=True, help="输出目录")
    run.add_argument("--config", default=None, help="YAML 配置文件")
    run.add_argument("--fan-chunksize", type=int, default=None, help="读取超大单风机 CSV 时的 chunksize，例如 2000000")

    # 常用覆盖项
    run.add_argument("--capacity-mw", type=float, default=None)
    run.add_argument("--expected-n-fans", type=int, default=None)
    run.add_argument("--normal-remove-vote-threshold", type=int, default=None)
    run.add_argument("--lowwind-remove-vote-threshold", type=int, default=None)
    run.add_argument("--disable-ae", action="store_true", help="关闭 Autoencoder 模块")
    run.add_argument("--enabled-methods", default=None, help="启用的方法，逗号分隔：adaptive_iqr,ransac_mad,isolation_forest,lof,autoencoder,low_power_belt")
    run.add_argument("--decision-mode", choices=["vote", "single", "any", "all", "weighted"], default=None, help="融合模式")
    run.add_argument("--single-method", default=None, help="decision-mode=single 时使用的方法")
    run.add_argument("--no-plots", action="store_true", help="不生成 PNG 图，加快批量对比")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        overrides = {
            "capacity_mw": args.capacity_mw,
            "expected_n_fans": args.expected_n_fans,
            "normal_remove_vote_threshold": args.normal_remove_vote_threshold,
            "lowwind_remove_vote_threshold": args.lowwind_remove_vote_threshold,
            "enabled_methods": args.enabled_methods,
            "decision_mode": args.decision_mode,
            "single_method": args.single_method,
            "make_plots": False if args.no_plots else None,
        }
        if args.disable_ae:
            overrides["ae_enabled"] = False
        cfg = load_config(args.config, overrides=overrides)
        summary = run_pipeline(
            qc_path=args.qc,
            mean_ws_path=args.mean_ws,
            fan_path=args.fan,
            out_dir=args.out,
            cfg=cfg,
            fan_chunksize=args.fan_chunksize,
        )
        print("清洗完成")
        print(f"输出目录: {Path(args.out).resolve()}")
        print(f"有效样本: {summary['valid_rows']}")
        print(f"保留样本: {summary['clean_rows']}")
        print(f"剔除样本: {summary['removed_rows']} ({summary['removed_rate_pct']:.2f}%)")
        print(f"低风速/切入过渡区终点: {summary['transition_end_ws']} m/s")
        print(f"接近额定区起点: {summary['rated_start_ws']} m/s")
        print(f"决策模式: {summary['decision_mode']}")
        print(f"启用方法: {', '.join(summary['enabled_methods'])}")


if __name__ == "__main__":
    main()

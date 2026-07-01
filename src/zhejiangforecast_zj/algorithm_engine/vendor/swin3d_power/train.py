from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from .config import MetSwin3DPowerConfig, Swin3DPowerConfig
from .data import ChannelStandardizer, NpyNwpPowerDataset, chronological_split_indices
from .modeling_met_swin3d_power import MetSwin3DRegressor
from .modeling_swin3d_power import NwpSwin3DRegressor
from .trainer import SimpleTrainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin3D NWP power regressor without transformers dependency.")
    parser.add_argument("--x", required=True, help="Path to .npy input shaped [N,C,S,H,W]")
    parser.add_argument("--y", required=True, help="Path to .npy label shaped [N] or [N,out_dim]")
    parser.add_argument("--output-dir", default="runs/met_swin3d_power")
    parser.add_argument("--model-kind", choices=["met_swin3d", "flat_swin3d"], default="met_swin3d")
    parser.add_argument("--model-config", default=None, help="Optional config.json")
    parser.add_argument("--in-chans", type=int, default=14)
    parser.add_argument("--out-dim", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--schema", choices=["derived14", "raw_uv14", "flat"], default="derived14")
    parser.add_argument("--num-pressure-levels", type=int, default=4)
    parser.add_argument("--max-lead-steps", type=int, default=16)
    parser.add_argument("--disable-residual-power-curve", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true", help="Enable activation checkpointing for Swin layers")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--capacity", type=float, default=None, help="Optional installed capacity for label scaling")
    parser.add_argument("--loss", choices=["mse", "mae", "huber"], default="huber")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--standardizer-chunk-size", type=int, default=2048, help="Chunk size for memmap scaler fitting")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    if args.model_config:
        if args.model_kind == "met_swin3d":
            cfg = MetSwin3DPowerConfig.load_json(args.model_config)
            return MetSwin3DRegressor(cfg), cfg
        cfg = Swin3DPowerConfig.load_json(args.model_config)
        return NwpSwin3DRegressor(cfg), cfg

    if args.model_kind == "met_swin3d":
        cfg = MetSwin3DPowerConfig(
            in_chans=args.in_chans,
            out_dim=args.out_dim,
            embed_dim=args.embed_dim,
            schema=args.schema,
            num_pressure_levels=args.num_pressure_levels,
            max_lead_steps=args.max_lead_steps,
            use_residual_power_curve=not args.disable_residual_power_curve,
            use_checkpoint=args.use_checkpoint,
            out_activation="sigmoid" if args.capacity is not None else "none",
        )
        return MetSwin3DRegressor(cfg), cfg

    cfg = Swin3DPowerConfig(
        in_chans=args.in_chans,
        out_dim=args.out_dim,
        embed_dim=args.embed_dim,
        out_activation="sigmoid" if args.capacity is not None else "none",
    )
    return NwpSwin3DRegressor(cfg), cfg


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_mem = np.load(args.x, mmap_mode="r")
    if x_mem.ndim != 5:
        raise ValueError(f"Expected x [N,C,S,H,W], got {x_mem.shape}")
    n = int(x_mem.shape[0])
    train_idx, val_idx, _ = chronological_split_indices(n, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # Fit channel stats on train period only to prevent leakage.
    # Use chunked memmap fitting so large HPC arrays are not materialized in RAM.
    standardizer = ChannelStandardizer.fit_memmap(
        x_mem, indices=train_idx, chunk_size=args.standardizer_chunk_size
    )
    standardizer.save_json(output_dir / "channel_standardizer.json")

    full_ds = NpyNwpPowerDataset(args.x, args.y, standardizer=standardizer, y_scale=args.capacity, mmap_mode="r")
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist()) if len(val_idx) else None

    model, cfg = build_model(args)
    cfg.save_json(output_dir / "config.json")
    print(f"model_kind={args.model_kind} parameters={model.count_parameters():,}")

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 32),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        fp16=args.fp16,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    trainer = SimpleTrainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=val_ds)
    metrics = trainer.train()
    print(metrics)


if __name__ == "__main__":
    main()

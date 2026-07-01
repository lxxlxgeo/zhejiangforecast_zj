"""Train LoRA-Swin3D or full-finetune Swin3D on synthetic NWP power data.

Examples
--------
PYTHONPATH=src python -m lora_swin3d_power.train \
  --config configs/tiny_lora_swin3d.json \
  --mode lora --epochs 2 --output-dir runs/demo_lora

PYTHONPATH=src python -m lora_swin3d_power.train \
  --config configs/tiny_lora_swin3d.json \
  --mode full --epochs 2 --output-dir runs/demo_full
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .data.synthetic import SyntheticNWPPowerDataset
from .models.build import prepare_full_finetune_model, prepare_lora_finetune_model
from .models.lora import print_trainable_parameter_summary, save_lora_adapter
from .training.losses import point_regression_loss
from .training.metrics import average_metric_dict, regression_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch], dim=0) for k in batch[0].keys()}


def make_loaders(cfg: ProjectConfig) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg.data
    train_ds = SyntheticNWPPowerDataset(
        num_samples=dcfg.num_samples,
        in_channels=dcfg.in_channels,
        lead_steps=dcfg.lead_steps,
        height=dcfg.height,
        width=dcfg.width,
        horizon=dcfg.horizon,
        capacity_min_mw=dcfg.capacity_min_mw,
        capacity_max_mw=dcfg.capacity_max_mw,
        noise_std=dcfg.noise_std,
        seed=dcfg.seed,
    )
    eval_ds = SyntheticNWPPowerDataset(
        num_samples=dcfg.eval_samples,
        in_channels=dcfg.in_channels,
        lead_steps=dcfg.lead_steps,
        height=dcfg.height,
        width=dcfg.width,
        horizon=dcfg.horizon,
        capacity_min_mw=dcfg.capacity_min_mw,
        capacity_max_mw=dcfg.capacity_max_mw,
        noise_std=dcfg.noise_std,
        seed=dcfg.seed + 100_000,
    )
    tcfg = cfg.training
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        shuffle=True,
        num_workers=tcfg.num_workers,
        pin_memory=False,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        num_workers=tcfg.num_workers,
        pin_memory=False,
        collate_fn=collate_batch,
    )
    return train_loader, eval_loader


def build_model_and_optimizer(cfg: ProjectConfig, base_checkpoint: str | None = None):
    tcfg = cfg.training
    device = torch.device(tcfg.device)
    if tcfg.mode == "full":
        model = prepare_full_finetune_model(cfg.model, base_checkpoint=base_checkpoint, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr_full, weight_decay=tcfg.weight_decay_full)
        replaced = []
    else:
        model, replaced, summary = prepare_lora_finetune_model(
            cfg.model,
            cfg.lora,
            base_checkpoint=base_checkpoint,
            device=device,
        )
        print("LoRA injected modules:")
        for name in replaced:
            print(f"  - {name}")
        print_trainable_parameter_summary(model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=tcfg.lr_lora, weight_decay=tcfg.weight_decay_lora)
    return model, optimizer, replaced


def train_one_epoch(model, loader, optimizer, cfg: ProjectConfig) -> dict[str, float]:
    model.train()
    device = torch.device(cfg.training.device)
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        capacity = batch["capacity_mw"].to(device)
        pred = model(x)
        if pred.ndim == 3 and y.ndim == 2:
            y = y.unsqueeze(1).expand_as(pred)
        loss = point_regression_loss(pred, y, loss_type=cfg.training.loss_type, ramp_weight=cfg.training.ramp_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        metric_items.append(regression_metrics(pred.detach(), y.detach(), capacity))

    metrics = average_metric_dict(metric_items)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, cfg: ProjectConfig) -> dict[str, float]:
    model.eval()
    device = torch.device(cfg.training.device)
    losses: list[float] = []
    metric_items: list[dict[str, float]] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        capacity = batch["capacity_mw"].to(device)
        pred = model(x)
        if pred.ndim == 3 and y.ndim == 2:
            y = y.unsqueeze(1).expand_as(pred)
        loss = point_regression_loss(pred, y, loss_type=cfg.training.loss_type, ramp_weight=cfg.training.ramp_weight)
        losses.append(float(loss.detach().cpu()))
        metric_items.append(regression_metrics(pred, y, capacity))
    metrics = average_metric_dict(metric_items)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    return metrics


def save_checkpoint(model, cfg: ProjectConfig, output_dir: Path, name: str, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "lora":
        save_lora_adapter(model, str(output_dir / f"{name}_adapter.pt"), include_keywords=cfg.lora.trainable_keywords)
    else:
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, output_dir / f"{name}_full.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny_lora_swin3d.json")
    parser.add_argument("--mode", choices=["lora", "full"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Optional full/base checkpoint before LoRA injection")
    args = parser.parse_args()

    cfg = ProjectConfig.load_json(args.config)
    if args.mode is not None:
        cfg.training.mode = args.mode
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.device is not None:
        cfg.training.device = args.device
    if args.output_dir is not None:
        cfg.training.output_dir = args.output_dir
    cfg.validate()

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(output_dir / "resolved_config.json")
    log_path = output_dir / "train_log.jsonl"

    set_seed(cfg.training.seed)
    train_loader, eval_loader = make_loaders(cfg)
    model, optimizer, replaced = build_model_and_optimizer(cfg, base_checkpoint=args.base_checkpoint)

    best_eval = float("inf")
    t0 = time.time()
    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg)
        eval_metrics = evaluate(model, eval_loader, cfg)
        row = {
            "epoch": epoch,
            "mode": cfg.training.mode,
            "train": train_metrics,
            "eval": eval_metrics,
            "elapsed_sec": round(time.time() - t0, 3),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch:03d} mode={cfg.training.mode} "
            f"train_loss={train_metrics['loss']:.5f} eval_loss={eval_metrics['loss']:.5f} "
            f"eval_mae_norm={eval_metrics['mae_norm']:.5f} eval_mae_mw={eval_metrics.get('mae_mw', 0.0):.3f}"
        )
        if eval_metrics["loss"] < best_eval:
            best_eval = eval_metrics["loss"]
            save_checkpoint(model, cfg, output_dir, "best", cfg.training.mode)
        if cfg.training.save_every_epoch:
            save_checkpoint(model, cfg, output_dir, f"epoch_{epoch:03d}", cfg.training.mode)

    save_checkpoint(model, cfg, output_dir, "last", cfg.training.mode)
    print(f"Done. Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

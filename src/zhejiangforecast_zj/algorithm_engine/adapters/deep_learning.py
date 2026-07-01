from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zhejiangforecast_zj.algorithm_engine.adapters.vendor_paths import enable_vendor_project
from zhejiangforecast_zj.core.jsonx import write_json


@dataclass
class DeepTrainResult:
    status: str
    model_type: str
    artifact_path: str
    metrics: dict[str, float]
    extra: dict[str, Any]


def _load_tensor_meta(meta_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(meta_path).read_text(encoding="utf-8"))


def train_met_swin3d(
    *,
    train_x: str | Path,
    train_y: str | Path,
    eval_x: str | Path,
    eval_y: str | Path,
    tensor_meta: str | Path,
    out_dir: str | Path,
    epochs: int = 2,
    batch_size: int = 8,
    device: str = "cpu",
) -> DeepTrainResult:
    """Train the modified short-term MetSwin3D model on real ETL tensors."""

    try:
        enable_vendor_project("swin3d")
        import torch
        from swin3d_power.config import MetSwin3DPowerConfig
        from swin3d_power.data import ChannelStandardizer, NpyNwpPowerDataset
        from swin3d_power.modeling_met_swin3d_power import MetSwin3DRegressor
        from swin3d_power.trainer import SimpleTrainer, TrainingArguments
    except Exception as exc:
        return DeepTrainResult("SKIPPED", "met_swin3d", "", {}, {"reason": "dependency_unavailable", "error": str(exc)})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_tensor_meta(tensor_meta)
    x_train = np.load(train_x, mmap_mode="r")
    train_indices = np.arange(len(x_train))
    standardizer = ChannelStandardizer.fit_memmap(x_train, train_indices, chunk_size=256)
    standardizer.save_json(out_dir / "channel_standardizer.json")
    capacity = meta.get("capacity_mw") or 1.0
    train_ds = NpyNwpPowerDataset(train_x, train_y, standardizer=standardizer, y_scale=None, mmap_mode="r")
    eval_ds = NpyNwpPowerDataset(eval_x, eval_y, standardizer=standardizer, y_scale=None, mmap_mode="r")
    cfg = MetSwin3DPowerConfig(
        in_chans=int(meta["shape"][1]),
        out_dim=1,
        embed_dim=24,
        depths=(1, 1, 1),
        num_heads=(3, 3, 6),
        window_sizes=((3, 4, 4), (3, 4, 4), (3, 2, 2)),
        out_activation="sigmoid" if capacity else "none",
    )
    model = MetSwin3DRegressor(cfg)
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=int(epochs),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size),
        learning_rate=2e-4,
        eval_steps=0,
        save_steps=0,
        early_stopping_patience=0,
    )
    # The referenced trainer chooses device from its constructor, not TrainingArguments.
    trainer = SimpleTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, device=device)
    metrics = trainer.train()
    pred_norm = trainer.predict(eval_ds).detach().cpu().numpy().reshape(-1).astype(float)
    true_norm = np.load(eval_y).reshape(-1).astype(float)
    artifact = out_dir / "best" / "pytorch_model.bin"
    if not artifact.exists():
        artifact = out_dir / "last" / "pytorch_model.bin"
    write_json(out_dir / "model_card.json", {"model_type": "met_swin3d", "tensor_meta": meta, "metrics": metrics})
    return DeepTrainResult(
        "TRAINED",
        "met_swin3d",
        str(artifact),
        metrics,
        {
            "model_card": str(out_dir / "model_card.json"),
            "eval_pred_norm": pred_norm.tolist(),
            "eval_true_norm": true_norm.tolist(),
        },
    )


class _NpyDictDataset:
    def __init__(self, x_path: str | Path, y_path: str | Path, capacity_mw: float | None = None):
        import torch

        self.torch = torch
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.capacity = float(capacity_mw or 1.0)
        if len(self.x) != len(self.y):
            raise ValueError("x/y length mismatch")

    def __len__(self) -> int:
        return int(len(self.x))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        x = np.asarray(self.x[idx], dtype=np.float32)
        y = np.asarray(self.y[idx], dtype=np.float32).reshape(-1)
        return {
            "x": self.torch.from_numpy(x),
            "y": self.torch.from_numpy(y),
            "capacity_mw": self.torch.tensor([self.capacity], dtype=self.torch.float32),
            "y_mw": self.torch.from_numpy(y * self.capacity),
        }


def train_lora_swin3d(
    *,
    train_x: str | Path,
    train_y: str | Path,
    eval_x: str | Path,
    eval_y: str | Path,
    tensor_meta: str | Path,
    out_dir: str | Path,
    base_checkpoint: str | None = None,
    epochs: int = 2,
    batch_size: int = 8,
    device: str = "cpu",
) -> DeepTrainResult:
    """Train LoRA-Swin3D on real ETL tensors using the existing LoRA utilities."""

    try:
        enable_vendor_project("lora_swin3d")
        import torch
        from torch.utils.data import DataLoader
        from lora_swin3d_power.config import ProjectConfig
        from lora_swin3d_power.models.build import prepare_lora_finetune_model
        from lora_swin3d_power.models.lora import save_lora_adapter
        from lora_swin3d_power.models.swin3d_v2 import Swin3DV2RegressionConfig
        from lora_swin3d_power.training.losses import point_regression_loss
        from lora_swin3d_power.training.metrics import average_metric_dict, regression_metrics
    except Exception as exc:
        return DeepTrainResult("SKIPPED", "lora_swin3d", "", {}, {"reason": "dependency_unavailable", "error": str(exc)})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_tensor_meta(tensor_meta)
    capacity = float(meta.get("capacity_mw") or 1.0)
    model_cfg = Swin3DV2RegressionConfig(
        in_channels=int(meta["shape"][1]),
        out_channels=1,
        dims=(16, 32, 64),
        depths=(1, 1),
        num_heads=(4, 4),
        window_sizes=((1, 2, 2), (1, 2, 2)),
        patch_size=(1, 2, 2),
        output_activation="sigmoid",
    )
    cfg = ProjectConfig(model=model_cfg)
    cfg.training.device = device
    cfg.training.epochs = int(epochs)
    cfg.training.batch_size = int(batch_size)
    cfg.training.output_dir = str(out_dir)
    prepared = prepare_lora_finetune_model(cfg.model, cfg.lora, base_checkpoint=base_checkpoint, device=device)
    if isinstance(prepared, tuple):
        model = prepared[0]
        lora_summary = prepared[2] if len(prepared) > 2 else {}
    else:
        model = prepared
        lora_summary = {}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.training.lr_lora)
    train_ds = _NpyDictDataset(train_x, train_y, capacity)
    eval_ds = _NpyDictDataset(eval_x, eval_y, capacity)

    def collate(batch):
        return {
            "x": torch.stack([b["x"] for b in batch]),
            "y": torch.stack([b["y"] for b in batch]),
            "capacity_mw": torch.stack([b["capacity_mw"] for b in batch]),
        }

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate)
    device_obj = torch.device(device)
    model.to(device_obj)
    best_metrics: dict[str, float] = {}
    best_loss = float("inf")
    log_rows = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(device_obj)
            y = batch["y"].to(device_obj)
            pred = model(x)
            loss = point_regression_loss(pred, y, loss_type=cfg.training.loss_type)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        metrics_items = []
        eval_losses = []
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                x = batch["x"].to(device_obj)
                y = batch["y"].to(device_obj)
                cap = batch["capacity_mw"].to(device_obj)
                pred = model(x)
                loss = point_regression_loss(pred, y, loss_type=cfg.training.loss_type)
                eval_losses.append(float(loss.detach().cpu()))
                metrics_items.append(regression_metrics(pred, y, capacity_mw=cap))
        metrics = average_metric_dict(metrics_items)
        metrics["loss"] = float(np.mean(eval_losses)) if eval_losses else float("nan")
        log_rows.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "eval": metrics})
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_metrics = metrics
            save_lora_adapter(model, str(out_dir / "best_adapter.pt"), include_keywords=cfg.lora.trainable_keywords)
    save_lora_adapter(model, str(out_dir / "last_adapter.pt"), include_keywords=cfg.lora.trainable_keywords)
    pred_rows = []
    true_rows = []
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["x"].to(device_obj)
            y = batch["y"].to(device_obj)
            pred_rows.append(model(x).detach().cpu().numpy().reshape(-1))
            true_rows.append(y.detach().cpu().numpy().reshape(-1))
    pred_norm = np.concatenate(pred_rows).astype(float) if pred_rows else np.array([], dtype=float)
    true_norm = np.concatenate(true_rows).astype(float) if true_rows else np.array([], dtype=float)
    write_json(out_dir / "train_log.json", log_rows)
    cfg.save_json(out_dir / "resolved_config.json")
    write_json(
        out_dir / "model_card.json",
        {"model_type": "lora_swin3d", "tensor_meta": meta, "metrics": best_metrics, "lora_summary": lora_summary},
    )
    return DeepTrainResult(
        "TRAINED",
        "lora_swin3d",
        str(out_dir / "best_adapter.pt"),
        best_metrics,
        {
            "model_card": str(out_dir / "model_card.json"),
            "eval_pred_norm": pred_norm.tolist(),
            "eval_true_norm": true_norm.tolist(),
        },
    )

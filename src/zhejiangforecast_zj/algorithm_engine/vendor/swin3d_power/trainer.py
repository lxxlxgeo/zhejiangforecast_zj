from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Optional
import json
import math
import time

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


LossName = Literal["mse", "mae", "huber"]


@dataclass
class TrainingArguments:
    """Small Trainer-like argument block without depending on transformers."""

    output_dir: str = "runs/swin3d_power"
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    loss: LossName = "huber"
    huber_delta: float = 0.05
    fp16: bool = False
    num_workers: int = 0
    seed: int = 42
    logging_steps: int = 20
    eval_steps: int = 100
    save_steps: int = 100
    early_stopping_patience: int = 10
    minimize_metric: str = "rmse"
    pin_memory: bool = True

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
            factor = self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * factor

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(name: LossName, huber_delta: float) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=huber_delta)
    raise ValueError(f"Unsupported loss={name}")


@torch.no_grad()
def regression_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    err = pred - target
    mse = torch.mean(err**2).item()
    mae = torch.mean(torch.abs(err)).item()
    rmse = math.sqrt(max(mse, 0.0))
    ss_res = torch.sum(err**2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


class SimpleTrainer:
    """Minimal HF-Trainer-style loop for regression.

    Features included: AdamW, warmup+cosine LR, gradient accumulation, gradient
    clipping, AMP on CUDA, periodic eval, checkpointing, early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        collate_fn: Optional[Callable] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        set_seed(args.seed)
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.args.save_json(self.output_dir / "training_args.json")

        self.loss_fn = build_loss(args.loss, args.huber_delta)
        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        train_batches = math.ceil(len(train_dataset) / args.per_device_train_batch_size)
        update_steps_per_epoch = math.ceil(train_batches / max(1, args.gradient_accumulation_steps))
        total_steps = max(1, update_steps_per_epoch * args.num_train_epochs)
        warmup_steps = int(round(total_steps * args.warmup_ratio))
        self.scheduler = WarmupCosineScheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(args.fp16 and self.device.type == "cuda"))

    def _loader(self, dataset: Dataset, train: bool) -> DataLoader:
        batch_size = self.args.per_device_train_batch_size if train else self.args.per_device_eval_batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.args.pin_memory and self.device.type == "cuda",
            drop_last=False,
        )

    def _save_checkpoint(self, name: str, metrics: Optional[dict[str, float]] = None) -> None:
        ckpt_dir = self.output_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / "pytorch_model.bin")
        if hasattr(self.model, "config") and hasattr(self.model.config, "save_json"):
            self.model.config.save_json(ckpt_dir / "config.json")
        if metrics is not None:
            with (ckpt_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

    def train(self) -> dict[str, float]:
        train_loader = self._loader(self.train_dataset, train=True)
        best_metric = float("inf")
        best_metrics: dict[str, float] = {}
        bad_eval_count = 0
        global_step = 0
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(1, self.args.num_train_epochs + 1):
            self.model.train()
            running_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                x, y = batch
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=bool(self.args.fp16 and self.device.type == "cuda")):
                    pred = self.model(x)
                    loss = self.loss_fn(pred, y)
                    loss_to_backprop = loss / max(1, self.args.gradient_accumulation_steps)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()
                running_loss += loss.item()

                if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader):
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        avg_loss = running_loss / max(1, step)
                        lr = self.scheduler.get_last_lr()[0]
                        print(f"epoch={epoch} step={global_step} train_loss={avg_loss:.6f} lr={lr:.3e}")

                    do_eval = self.eval_dataset is not None and self.args.eval_steps > 0 and global_step % self.args.eval_steps == 0
                    if do_eval:
                        metrics = self.evaluate(prefix="eval")
                        current = metrics.get(f"eval_{self.args.minimize_metric}", float("inf"))
                        print("eval", metrics)
                        if current < best_metric:
                            best_metric = current
                            best_metrics = metrics
                            bad_eval_count = 0
                            self._save_checkpoint("best", metrics)
                        else:
                            bad_eval_count += 1
                        if self.args.early_stopping_patience > 0 and bad_eval_count >= self.args.early_stopping_patience:
                            print("early stopping")
                            return best_metrics

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"checkpoint-{global_step}")

            if self.eval_dataset is not None:
                metrics = self.evaluate(prefix="eval")
                current = metrics.get(f"eval_{self.args.minimize_metric}", float("inf"))
                print(f"epoch={epoch} done", metrics)
                if current < best_metric:
                    best_metric = current
                    best_metrics = metrics
                    bad_eval_count = 0
                    self._save_checkpoint("best", metrics)
                else:
                    bad_eval_count += 1
                if self.args.early_stopping_patience > 0 and bad_eval_count >= self.args.early_stopping_patience:
                    print("early stopping")
                    break

        final_metrics = self.evaluate(prefix="eval") if self.eval_dataset is not None else {}
        final_metrics["train_runtime_sec"] = time.time() - t0
        self._save_checkpoint("last", final_metrics)
        return best_metrics or final_metrics

    @torch.no_grad()
    def evaluate(self, prefix: str = "eval") -> dict[str, float]:
        if self.eval_dataset is None:
            return {}
        self.model.eval()
        loader = self._loader(self.eval_dataset, train=False)
        preds = []
        targets = []
        losses = []
        for batch in loader:
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
            losses.append(loss.item())
        pred = torch.cat(preds, dim=0)
        target = torch.cat(targets, dim=0)
        metrics = regression_metrics(pred, target)
        metrics[f"{prefix}_loss"] = float(sum(losses) / max(1, len(losses)))
        return {f"{prefix}_{k}": float(v) for k, v in metrics.items() if not k.startswith(prefix)} | {
            k: v for k, v in metrics.items() if k.startswith(prefix)
        }

    @torch.no_grad()
    def predict(self, dataset: Dataset) -> Tensor:
        self.model.eval()
        loader = self._loader(dataset, train=False)
        preds = []
        for x, _ in loader:
            x = x.to(self.device, non_blocking=True)
            preds.append(self.model(x).detach().cpu())
        return torch.cat(preds, dim=0)

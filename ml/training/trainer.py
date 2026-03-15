from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.config import Config
from ml.models.two_tower import TwoTowerModel
from ml.training.callbacks import EarlyStopping, build_scheduler
from ml.training.checkpoint import CheckpointManager
from ml.evaluation.evaluator import Evaluator
from ml.utils.logging import MetricLogger

logger = logging.getLogger(__name__)


class HistoryCollator:
    def __init__(
        self,
        user_histories: Dict[int, List[Tuple[int, float]]],
        max_history: int = 50,
    ):
        self.histories = user_histories
        self.max_history = max_history

    def __call__(self, batch):
        user_idxs, pos_idxs, neg_idxs = zip(*batch)
        B = len(user_idxs)
        S = self.max_history

        hist_items = torch.zeros(B, S, dtype=torch.long)
        hist_ratings = torch.zeros(B, S, dtype=torch.float)
        hist_mask = torch.zeros(B, S, dtype=torch.bool)

        for i, uidx in enumerate(user_idxs):
            hist = self.histories.get(uidx, [])
            hist = hist[-S:]
            for j, (item_idx, rating) in enumerate(hist):
                hist_items[i, j] = item_idx
                hist_ratings[i, j] = rating
                hist_mask[i, j] = True

        return (
            hist_items,
            hist_ratings,
            hist_mask,
            torch.tensor(pos_idxs, dtype=torch.long),
            torch.tensor(neg_idxs, dtype=torch.long),
        )


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: TwoTowerModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        evaluator: Optional[Evaluator],
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        self.optimizer = self._build_optimizer()

        num_steps = len(train_loader) * cfg.training.epochs
        self.scheduler = build_scheduler(self.optimizer, cfg.training.scheduler, num_steps)

        self.use_amp = cfg.training.mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.early_stopping = EarlyStopping(cfg.training.early_stopping)
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=cfg.paths.checkpoints,
            keep_top_k=cfg.training.checkpoint.keep_top_k,
            metric_name=cfg.training.early_stopping.metric,
        )

        self.metric_logger = MetricLogger(
            backend=cfg.logging.backend,
            project=cfg.logging.wandb_project,
            config=self._config_dict(),
            log_dir=cfg.paths.logs,
        )

        self.global_step = 0
        self.start_epoch = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        tcfg = self.cfg.training
        if tcfg.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=tcfg.learning_rate,
                weight_decay=tcfg.weight_decay,
            )
        elif tcfg.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=tcfg.learning_rate,
            )
        raise ValueError(f"Unknown optimizer: {tcfg.optimizer}")

    def _config_dict(self) -> dict:
        import dataclasses
        def _flat(obj, prefix=""):
            out = {}
            if dataclasses.is_dataclass(obj):
                for f in dataclasses.fields(obj):
                    out.update(_flat(getattr(obj, f.name), f"{prefix}{f.name}/"))
            elif isinstance(obj, (list, tuple)):
                out[prefix.rstrip("/")] = str(obj)
            else:
                out[prefix.rstrip("/")] = obj
            return out
        return _flat(self.cfg)

    def maybe_resume(self):
        ckpt = self.ckpt_manager.load_latest(self.device)
        if ckpt is None:
            logger.info("No checkpoint found, training from scratch")
            return
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from epoch %d (metric=%.4f)", ckpt["epoch"], ckpt["metric_value"])

    def train(self):
        logger.info("Starting training for %d epochs on %s", self.cfg.training.epochs, self.device)
        self.maybe_resume()

        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)

            val_metrics = {}
            primary_metric = 0.0
            if self.evaluator is not None and self.val_loader is not None:
                val_metrics = self.evaluator.evaluate(self.model, self.device)
                primary_metric = val_metrics.get(self.cfg.training.early_stopping.metric, 0.0)

            epoch_time = time.time() - t0

            log_dict = {
                "epoch": epoch,
                "epoch_time": epoch_time,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            log_dict.update({f"train/{k}": v for k, v in train_metrics.items()})
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            self.metric_logger.log(log_dict, step=epoch)

            logger.info(
                "Epoch %d/%d | loss=%.4f | %s=%.4f | time=%.1fs",
                epoch + 1, self.cfg.training.epochs,
                train_metrics.get("loss", 0),
                self.cfg.training.early_stopping.metric,
                primary_metric,
                epoch_time,
            )

            if (epoch + 1) % self.cfg.training.checkpoint.save_every == 0:
                self.ckpt_manager.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metric_value=primary_metric,
                )

            if self.early_stopping.step(primary_metric):
                logger.info("Training stopped early at epoch %d", epoch + 1)
                break

        self.metric_logger.finish()
        logger.info("Training complete!")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        metrics_acc = defaultdict(float)
        num_batches = 0

        for batch in self.train_loader:
            hist_items, hist_ratings, hist_mask, pos_idx, neg_idx = [
                t.to(self.device) for t in batch
            ]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    user_history_items=hist_items,
                    user_history_ratings=hist_ratings,
                    user_history_mask=hist_mask,
                    pos_item_indices=pos_idx,
                    neg_item_indices=neg_idx,
                )
                loss = outputs["loss"]

            self.scaler.scale(loss).backward()

            if self.cfg.training.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.gradient_clip
                )
            else:
                grad_norm = 0.0

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            metrics_acc["loss"] += loss.item()
            metrics_acc["pos_score"] += outputs["pos_score"].item()
            metrics_acc["neg_score"] += outputs["neg_score"].item()
            metrics_acc["margin"] += outputs["margin"].item()
            metrics_acc["grad_norm"] += float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            num_batches += 1

            self.global_step += 1
            if self.global_step % self.cfg.logging.log_every_n_steps == 0:
                self.metric_logger.log(
                    {
                        "step/loss": loss.item(),
                        "step/lr": self.optimizer.param_groups[0]["lr"],
                        "step/grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    },
                    step=self.global_step,
                )

        return {k: v / max(num_batches, 1) for k, v in metrics_acc.items()}

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau, StepLR

from ml.config import SchedulerConfig, EarlyStoppingConfig

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, cfg: EarlyStoppingConfig, higher_is_better: bool = True):
        self.patience = cfg.patience
        self.min_delta = cfg.min_delta
        self.metric_name = cfg.metric
        self.higher_is_better = higher_is_better

        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric_value: float) -> bool:
        if self.best_value is None:
            self.best_value = metric_value
            return False

        if self.higher_is_better:
            improved = metric_value > self.best_value + self.min_delta
        else:
            improved = metric_value < self.best_value - self.min_delta

        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered: %s did not improve for %d epochs (best=%.4f)",
                    self.metric_name, self.patience, self.best_value,
                )
                return True

        return False


def build_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = int(num_training_steps * cfg.warmup_ratio)

    if cfg.type == "cosine_warmup":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return max(cfg.eta_min / 1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    elif cfg.type == "step":
        return StepLR(optimizer, step_size=num_training_steps // 3, gamma=0.1)

    elif cfg.type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}")

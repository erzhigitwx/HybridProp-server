from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MetricLogger:
    def __init__(
        self,
        backend: str = "wandb",
        project: str = "hybridprop-recsys",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_dir: str = "logs",
    ):
        self.backend = backend
        self._wandb = None
        self._tb_writer = None

        if backend in ("wandb", "both"):
            try:
                import wandb

                wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                    reinit=True,
                )
                self._wandb = wandb
                logger.info("W&B run initialized: %s", wandb.run.name)
            except Exception as e:
                logger.warning("W&B init failed (%s), falling back to TensorBoard", e)
                self.backend = "tensorboard"

        if self.backend in ("tensorboard", "both"):
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = Path(log_dir) / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(str(tb_dir))
            logger.info("TensorBoard writer at %s", tb_dir)

    def log(self, metrics: Dict[str, float], step: int):
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, global_step=step)

    def log_table(self, name: str, columns: list, data: list, step: int):
        if self._wandb is not None:
            table = self._wandb.Table(columns=columns, data=data)
            self._wandb.log({name: table}, step=step)

    def finish(self):
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()

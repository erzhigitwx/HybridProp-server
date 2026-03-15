from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointEntry:
    path: Path
    epoch: int
    metric: float


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        keep_top_k: int = 3,
        metric_name: str = "ndcg@10",
        higher_is_better: bool = True,
    ):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.entries: List[CheckpointEntry] = []

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metric_value: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = self.dir / f"checkpoint_epoch{epoch:03d}_{metric_value:.4f}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_name": self.metric_name,
            "metric_value": metric_value,
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if extra:
            state.update(extra)

        torch.save(state, path)
        self.entries.append(CheckpointEntry(path=path, epoch=epoch, metric=metric_value))
        logger.info("Saved checkpoint: %s (%.4f)", path.name, metric_value)

        latest = self.dir / "latest.pt"
        shutil.copy2(path, latest)

        self._prune()

        best = self._best_entry()
        if best is not None:
            best_path = self.dir / "best.pt"
            shutil.copy2(best.path, best_path)

        return path

    def load_best(self, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
        best_path = self.dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint at {best_path}")
        return torch.load(best_path, map_location=device, weights_only=False)

    def load_latest(self, device: torch.device = torch.device("cpu")) -> Optional[Dict[str, Any]]:
        latest_path = self.dir / "latest.pt"
        if not latest_path.exists():
            return None
        return torch.load(latest_path, map_location=device, weights_only=False)

    def _best_entry(self) -> Optional[CheckpointEntry]:
        if not self.entries:
            return None
        return sorted(
            self.entries,
            key=lambda e: e.metric,
            reverse=self.higher_is_better,
        )[0]

    def _prune(self):
        if len(self.entries) <= self.keep_top_k:
            return
        sorted_entries = sorted(
            self.entries,
            key=lambda e: e.metric,
            reverse=self.higher_is_better,
        )
        keep = set(id(e) for e in sorted_entries[: self.keep_top_k])
        new_entries = []
        for entry in self.entries:
            if id(entry) in keep:
                new_entries.append(entry)
            else:
                if entry.path.exists():
                    entry.path.unlink()
                    logger.debug("Pruned checkpoint: %s", entry.path.name)
        self.entries = new_entries

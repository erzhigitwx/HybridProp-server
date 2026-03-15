from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
from ml.config import TowerConfig


class RestaurantTower(nn.Module):
    def __init__(self, cfg: TowerConfig):
        super().__init__()
        self.cfg = cfg
        layers: List[nn.Module] = []
        in_dim = cfg.input_dim

        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(cfg.activation))
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, cfg.output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.mlp(features)
        out = nn.functional.normalize(out, dim=-1)
        return out

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }.get(name.lower(), nn.GELU())

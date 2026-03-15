from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from ml.config import ModelConfig
from ml.models.restaurant_tower import RestaurantTower
from ml.models.user_tower import UserTower
from ml.models.losses import build_loss


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        item_features: np.ndarray,
        user_histories: Optional[Dict] = None,
    ):
        super().__init__()
        self.cfg = cfg
        num_items, raw_dim = item_features.shape

        tower_cfg = cfg.restaurant_tower
        if tower_cfg.input_dim != raw_dim:
            tower_cfg.input_dim = raw_dim

        self.restaurant_tower = RestaurantTower(tower_cfg)
        self.user_tower = UserTower(cfg.user_tower, item_dim=tower_cfg.output_dim)

        self.loss_fn = build_loss(cfg.loss.type, cfg.loss.infonce_temperature)

        self.register_buffer(
            "item_features",
            torch.tensor(item_features, dtype=torch.float32),
        )

        self._item_emb_cache: Optional[torch.Tensor] = None

    @property
    def item_dim(self) -> int:
        return self.cfg.restaurant_tower.output_dim

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        features = self.item_features[item_indices]
        return self.restaurant_tower(features)

    @torch.no_grad()
    def encode_all_items(self, batch_size: int = 1024) -> torch.Tensor:
        self.restaurant_tower.eval()
        embs = []
        N = self.item_features.shape[0]
        for i in range(0, N, batch_size):
            chunk = self.item_features[i : i + batch_size]
            emb = self.restaurant_tower(chunk)
            embs.append(emb.cpu())
        return torch.cat(embs, dim=0)

    def encode_user(
        self,
        history_item_indices: torch.Tensor,  # (B, S)
        history_ratings: torch.Tensor,        # (B, S)
        history_mask: torch.Tensor,           # (B, S)
    ) -> torch.Tensor:
        B, S = history_item_indices.shape

        flat_indices = history_item_indices.reshape(-1)

        with torch.no_grad():
            flat_embs = self.encode_items(flat_indices)  # (B*S, D)

        hist_embs = flat_embs.view(B, S, -1)  # (B, S, D)
        return self.user_tower(hist_embs, history_ratings, history_mask)

    def forward(
        self,
        user_history_items: torch.Tensor,   # (B, S)
        user_history_ratings: torch.Tensor,  # (B, S)
        user_history_mask: torch.Tensor,     # (B, S)
        pos_item_indices: torch.Tensor,      # (B,)
        neg_item_indices: torch.Tensor,      # (B,)
    ) -> Dict[str, torch.Tensor]:
        user_emb = self.encode_user(user_history_items, user_history_ratings, user_history_mask)
        pos_emb = self.encode_items(pos_item_indices)
        neg_emb = self.encode_items(neg_item_indices)

        loss = self.loss_fn(user_emb, pos_emb, neg_emb)

        with torch.no_grad():
            pos_score = (user_emb * pos_emb).sum(dim=-1).mean()
            neg_score = (user_emb * neg_emb).sum(dim=-1).mean()

        return {
            "loss": loss,
            "pos_score": pos_score,
            "neg_score": neg_score,
            "margin": pos_score - neg_score,
        }
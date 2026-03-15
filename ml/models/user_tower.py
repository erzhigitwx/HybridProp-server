from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import UserTowerConfig

class UserTower(nn.Module):
    def __init__(self, cfg: UserTowerConfig, item_dim: int):
        super().__init__()
        self.cfg = cfg
        self.item_dim = item_dim

        self.item_proj = nn.Linear(item_dim + 1, cfg.output_dim)
        self.query = nn.Parameter(torch.randn(1, 1, cfg.output_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.output_dim,
            num_heads=cfg.attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(cfg.output_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.output_dim, cfg.output_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.output_dim, cfg.output_dim),
        )

    def forward(
        self,
        item_embs: torch.Tensor,
        ratings: torch.Tensor,       # (B, S) float 1-5
        mask: torch.Tensor,          # (B, S) bool: True=valid
    ) -> torch.Tensor:
        B, S, _ = item_embs.shape

        ratings_feat = ratings.unsqueeze(-1) / 5.0
        x = torch.cat([item_embs, ratings_feat], dim=-1)  # (B, S, item_dim+1)
        x = self.item_proj(x)  # (B, S, D)

        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        key_padding_mask = ~mask
        attn_out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)  # (B, 1, D)
        attn_out = attn_out.squeeze(1)  # (B, D)

        out = self.layer_norm(attn_out)
        out = self.out_proj(out) + out
        out = F.normalize(out, dim=-1)
        return out

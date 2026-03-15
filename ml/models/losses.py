from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def forward(
        self,
        user_emb: torch.Tensor,    # (B, D)
        pos_emb: torch.Tensor,     # (B, D)
        neg_emb: torch.Tensor,     # (B, D)
    ) -> torch.Tensor:
        pos_score = (user_emb * pos_emb).sum(dim=-1)
        neg_score = (user_emb * neg_emb).sum(dim=-1)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        user_emb: torch.Tensor,    # (B, D)
        pos_emb: torch.Tensor,     # (B, D)
        neg_emb: torch.Tensor,     # (B, D)
    ) -> torch.Tensor:
        B = user_emb.shape[0]

        user_emb = F.normalize(user_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_scores = (user_emb * pos_emb).sum(dim=-1) / self.temperature

        all_items = torch.cat([pos_emb, neg_emb], dim=0)  # (2B, D)
        logits = user_emb @ all_items.T / self.temperature  # (B, 2B)

        labels = torch.arange(B, device=user_emb.device)
        loss = F.cross_entropy(logits, labels)
        return loss


def build_loss(loss_type: str, temperature: float = 0.07) -> nn.Module:
    if loss_type == "bpr":
        return BPRLoss()
    elif loss_type == "infonce":
        return InfoNCELoss(temperature=temperature)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

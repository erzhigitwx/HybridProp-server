from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ml.config import Config
from ml.models.two_tower import TwoTowerModel
from ml.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(
        self,
        cfg: Config,
        eval_users: List[Tuple[int, Set[int]]],
        user_histories: Dict[int, List[Tuple[int, float]]],
        train_interactions: Dict[int, Set[int]],
        max_history: int = 50,
    ):
        self.cfg = cfg
        self.eval_users = eval_users
        self.user_histories = user_histories
        self.train_interactions = train_interactions
        self.max_history = max_history
        self.ks = cfg.evaluation.ks

    @torch.no_grad()
    def evaluate(
        self,
        model: TwoTowerModel,
        device: torch.device,
        max_users: Optional[int] = None,
    ) -> Dict[str, float]:
        model.eval()
        max_users = max_users or self.cfg.evaluation.num_eval_users

        all_item_embs = model.encode_all_items(batch_size=1024)  # (N, D)
        all_item_embs = all_item_embs.to(device)
        N = all_item_embs.shape[0]

        max_k = max(self.ks)
        users_to_eval = self.eval_users[:max_users]
        all_ranked = []
        all_gt = []

        for user_idx, gt_items in tqdm(users_to_eval, desc="Evaluating", leave=False):
            hist = self.user_histories.get(user_idx, [])
            hist = hist[-self.max_history:]
            S = len(hist)
            if S == 0:
                continue

            hist_items = torch.zeros(1, self.max_history, dtype=torch.long, device=device)
            hist_ratings = torch.zeros(1, self.max_history, dtype=torch.float, device=device)
            hist_mask = torch.zeros(1, self.max_history, dtype=torch.bool, device=device)

            for j, (item_idx, rating) in enumerate(hist):
                hist_items[0, j] = item_idx
                hist_ratings[0, j] = rating
                hist_mask[0, j] = True

            user_emb = model.encode_user(hist_items, hist_ratings, hist_mask)  # (1, D)

            scores = (user_emb @ all_item_embs.T).squeeze(0)  # (N,)

            train_items = self.train_interactions.get(user_idx, set())
            for idx in train_items:
                if idx < N:
                    scores[idx] = -1e9

            _, top_indices = torch.topk(scores, max_k)
            all_ranked.append(top_indices.cpu().numpy())
            all_gt.append(gt_items)

        if not all_ranked:
            return {f"{m}@{k}": 0.0 for k in self.ks for m in ["hit_rate", "ndcg", "mrr"]}

        ranked_matrix = np.stack(all_ranked)  # (num_users, max_k)
        metrics = compute_all_metrics(ranked_matrix, all_gt, self.ks)

        model.train()
        return metrics

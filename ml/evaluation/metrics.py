from __future__ import annotations

import numpy as np
from typing import List


def hit_rate_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[set],
    k: int,
) -> float:
    hits = 0
    for i, gt in enumerate(ground_truth):
        if not gt:
            continue
        top_k = set(ranked_items[i, :k].tolist())
        if top_k & gt:
            hits += 1
    return hits / max(len(ground_truth), 1)


def ndcg_at_k(
    ranked_items: np.ndarray,
    ground_truth: List[set],
    k: int,
) -> float:
    ndcgs = []
    for i, gt in enumerate(ground_truth):
        if not gt:
            continue
        dcg = 0.0
        for rank in range(min(k, ranked_items.shape[1])):
            if ranked_items[i, rank] in gt:
                dcg += 1.0 / np.log2(rank + 2)
        ideal_dcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), k)))
        ndcgs.append(dcg / max(ideal_dcg, 1e-10))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def mrr(
    ranked_items: np.ndarray,
    ground_truth: List[set],
    k: int,
) -> float:
    rrs = []
    for i, gt in enumerate(ground_truth):
        if not gt:
            continue
        rr = 0.0
        for rank in range(min(k, ranked_items.shape[1])):
            if ranked_items[i, rank] in gt:
                rr = 1.0 / (rank + 1)
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def compute_all_metrics(
    ranked_items: np.ndarray,
    ground_truth: List[set],
    ks: List[int],
) -> dict:
    results = {}
    max_k = max(ks)
    for k in ks:
        results[f"hit_rate@{k}"] = hit_rate_at_k(ranked_items, ground_truth, k)
        results[f"ndcg@{k}"] = ndcg_at_k(ranked_items, ground_truth, k)
        results[f"mrr@{k}"] = mrr(ranked_items, ground_truth, k)
    return results

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ml.config import SoftFilterConfig

logger = logging.getLogger(__name__)


@dataclass
class FilterSpec:
    city: Optional[str] = None
    state: Optional[str] = None
    stars_min: Optional[float] = None
    stars_max: Optional[float] = None
    price_range_min: Optional[int] = None
    price_range_max: Optional[int] = None
    categories: Optional[List[str]] = None


@dataclass
class ScoredResult:
    business_id: str
    name: str
    score: float
    filter_match: float
    payload: Dict[str, Any] = field(default_factory=dict)
    soft_reasons: List[str] = field(default_factory=list)


class SoftFilterEngine:

    def __init__(self, cfg: SoftFilterConfig, qdrant_client, collection_name: str):
        self.cfg = cfg
        self.client = qdrant_client
        self.collection = collection_name

    def search(
        self,
        query_vector: np.ndarray,
        filters: FilterSpec,
        top_k: int = 20,
        softness: Optional[float] = None,
        excluded_ids: Optional[Set[str]] = None,
    ) -> List[ScoredResult]:
        softness = softness if softness is not None else self.cfg.default_softness
        excluded_ids = excluded_ids or set()

        if softness <= 0.01:
            strict_filter = self._build_qdrant_filter(filters)
            results = self._qdrant_search(query_vector, strict_filter, top_k + len(excluded_ids))
            return self._to_scored(results, filters, excluded_ids)[:top_k]

        relaxed_filters = self._relax_filters(filters, softness)
        relaxed_filter = self._build_qdrant_filter(relaxed_filters)
        fetch_count = top_k * 4 + len(excluded_ids)
        all_results = self._qdrant_search(query_vector, relaxed_filter, fetch_count)

        scored = []
        for r in all_results:
            biz_id = r.payload.get("business_id", "")
            if biz_id in excluded_ids:
                continue

            base_score = r.score
            match_score, reasons = self._compute_filter_match(r.payload, filters)

            bonus = self.cfg.filter_bonus * match_score
            final_score = base_score + bonus

            scored.append(ScoredResult(
                business_id=biz_id,
                name=r.payload.get("name", ""),
                score=final_score,
                filter_match=match_score,
                payload=r.payload,
                soft_reasons=reasons,
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _to_scored(self, results, filters: FilterSpec, excluded_ids: Set[str]) -> List[ScoredResult]:
        scored = []
        for r in results:
            biz_id = r.payload.get("business_id", "")
            if biz_id in excluded_ids:
                continue
            match_score, reasons = self._compute_filter_match(r.payload, filters)
            scored.append(ScoredResult(
                business_id=biz_id,
                name=r.payload.get("name", ""),
                score=r.score,
                filter_match=match_score,
                payload=r.payload,
                soft_reasons=reasons,
            ))
        return scored

    def _build_qdrant_filter(self, filters: FilterSpec):
        from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

        conditions = []
        if filters.city:
            conditions.append(FieldCondition(key="city", match=MatchValue(value=filters.city)))
        if filters.state:
            conditions.append(FieldCondition(key="state", match=MatchValue(value=filters.state)))
        if filters.stars_min is not None:
            conditions.append(FieldCondition(key="stars", range=Range(gte=filters.stars_min)))
        if filters.stars_max is not None:
            conditions.append(FieldCondition(key="stars", range=Range(lte=filters.stars_max)))
        if filters.price_range_min is not None:
            conditions.append(FieldCondition(key="price_range", range=Range(gte=filters.price_range_min)))
        if filters.price_range_max is not None:
            conditions.append(FieldCondition(key="price_range", range=Range(lte=filters.price_range_max)))

        if not conditions:
            return None
        return Filter(must=conditions)

    def _relax_filters(self, filters: FilterSpec, softness: float) -> FilterSpec:
        relaxed = FilterSpec(
            city=filters.city,
            state=filters.state,
            categories=filters.categories,
        )

        if filters.stars_min is not None:
            relaxed.stars_min = max(1.0, filters.stars_min - (softness * 2.0))
        if filters.stars_max is not None:
            relaxed.stars_max = min(5.0, filters.stars_max + (softness * 2.0))
        if filters.price_range_min is not None:
            relaxed.price_range_min = max(1, filters.price_range_min - (1 if softness > 0.1 else 0))
        if filters.price_range_max is not None:
            relaxed.price_range_max = min(4, filters.price_range_max + (1 if softness > 0.1 else 0))

        return relaxed

    def _compute_filter_match(
        self,
        payload: Dict[str, Any],
        original_filters: FilterSpec,
    ) -> Tuple[float, List[str]]:
        penalties = []
        reasons = []

        if original_filters.stars_min is not None:
            actual = payload.get("stars", 0)
            if actual < original_filters.stars_min:
                gap = original_filters.stars_min - actual
                penalty = min(gap / 2.0, 0.5)
                penalties.append(penalty)
                reasons.append(f"{actual}★ (filter: {original_filters.stars_min}+)")

        if original_filters.stars_max is not None:
            actual = payload.get("stars", 0)
            if actual > original_filters.stars_max:
                gap = actual - original_filters.stars_max
                penalty = min(gap / 2.0, 0.5)
                penalties.append(penalty)
                reasons.append(f"{actual}★ (filter: ≤{original_filters.stars_max})")

        if original_filters.price_range_max is not None:
            actual = payload.get("price_range", 0)
            if actual > original_filters.price_range_max:
                gap = actual - original_filters.price_range_max
                penalty = min(gap / 3.0, 0.4)
                penalties.append(penalty)
                reasons.append(f"Price {actual} (filter: ≤{original_filters.price_range_max})")

        if not penalties:
            return 1.0, []

        total_penalty = min(sum(penalties), 0.8)
        return round(1.0 - total_penalty, 2), reasons

    def _qdrant_search(self, vector: np.ndarray, filter_obj, limit: int):
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            query_filter=filter_obj,
            limit=limit,
        ).points
        return results
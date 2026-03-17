from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ml.config import Config
from ml.inference.soft_filter import SoftFilterEngine, FilterSpec, ScoredResult
from ml.utils.device import get_device

logger = logging.getLogger(__name__)

DISLIKE_THRESHOLD = 2.5


class Recommender:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self._model = None
        self._qdrant_client = None
        self._soft_filter: Optional[SoftFilterEngine] = None
        self._item_embeddings: Optional[np.ndarray] = None
        self._business_ids: Optional[List[str]] = None
        self._biz_id_to_idx: Optional[Dict[str, int]] = None

    def load(self):
        self._load_model()
        self._load_item_data()
        self._connect_qdrant()
        logger.info("Recommender ready: %d restaurants indexed", len(self._business_ids))

    def _load_model(self):
        from ml.training.checkpoint import CheckpointManager
        from ml.embeddings.aggregator import load_embeddings
        from ml.models.two_tower import TwoTowerModel

        features, ids = load_embeddings(self.cfg.paths.embeddings)
        self._business_ids = ids
        self._biz_id_to_idx = {bid: i for i, bid in enumerate(ids)}

        self._model = TwoTowerModel(self.cfg.model, features)
        ckpt_mgr = CheckpointManager(self.cfg.paths.checkpoints)
        ckpt = ckpt_mgr.load_best(self.device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.to(self.device)
        self._model.eval()
        logger.info("Loaded model from best checkpoint (epoch=%d)", ckpt["epoch"])

    def _connect_qdrant(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
        import pandas as pd
        from pathlib import Path

        logger.info("Initializing embedded Qdrant (in-memory)...")
        self._qdrant_client = QdrantClient(":memory:")

        collection = self.cfg.qdrant.collection_name
        self._qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=self.cfg.qdrant.vector_dim,
                distance=Distance.COSINE,
            ),
        )

        processed = Path(self.cfg.paths.processed)
        biz_lookup = {}
        if (processed / "businesses.parquet").exists():
            businesses = pd.read_parquet(processed / "businesses.parquet")
            biz_lookup = businesses.set_index("business_id").to_dict("index")

        points = []
        for idx, (biz_id, emb) in enumerate(zip(self._business_ids, self._item_embeddings)):
            meta = biz_lookup.get(biz_id, {})
            payload = {
                "business_id": biz_id,
                "name": meta.get("name", ""),
                "city": meta.get("city", ""),
                "state": meta.get("state", ""),
                "stars": float(meta.get("stars", 0)),
                "review_count": int(meta.get("review_count", 0)),
                "price_range": int(meta.get("price_range", 0)) if pd.notna(meta.get("price_range")) else 0,
                "categories": meta.get("categories", ""),
                "latitude": float(meta.get("latitude", 0)),
                "longitude": float(meta.get("longitude", 0)),
            }
            points.append(PointStruct(id=idx, vector=emb.tolist(), payload=payload))

        self._qdrant_client.upsert(collection_name=collection, points=points)
        logger.info("Embedded Qdrant loaded: %d points", len(points))

        self._soft_filter = SoftFilterEngine(
            cfg=self.cfg.api.soft_filter,
            qdrant_client=self._qdrant_client,
            collection_name=collection,
        )

    def _load_item_data(self):
        with torch.no_grad():
            embs = self._model.encode_all_items(batch_size=1024)
        self._item_embeddings = embs.numpy()

    def _get_excluded_ids(self, user_history: List[Tuple[str, float]]) -> Set[str]:
        return {bid for bid, rating in user_history if rating <= DISLIKE_THRESHOLD}

    @torch.no_grad()
    def recommend_with_soft_filter(
        self,
        user_history: List[Tuple[str, float]],
        filters: FilterSpec,
        top_k: int = 20,
        softness: Optional[float] = None,
    ) -> List[ScoredResult]:
        user_emb = self._encode_user_from_history(user_history)
        excluded = self._get_excluded_ids(user_history)
        return self._soft_filter.search(
            query_vector=user_emb,
            filters=filters,
            top_k=top_k,
            softness=softness,
            excluded_ids=excluded,
        )

    @torch.no_grad()
    def find_similar(
        self,
        business_id: str,
        top_k: int = 10,
        exclude_same: bool = True,
    ) -> List[ScoredResult]:
        idx = self._biz_id_to_idx.get(business_id)
        if idx is None:
            return []

        query_vector = self._item_embeddings[idx]
        results = self._qdrant_client.query_points(
            collection_name=self.cfg.qdrant.collection_name,
            query=query_vector.tolist(),
            limit=top_k + (1 if exclude_same else 0),
        ).points

        scored = []
        for r in results:
            biz = r.payload.get("business_id", "")
            if exclude_same and biz == business_id:
                continue
            scored.append(ScoredResult(
                business_id=biz,
                name=r.payload.get("name", ""),
                score=r.score,
                filter_match=1.0,
                payload=r.payload,
            ))
        return scored[:top_k]

    @torch.no_grad()
    def personalized_feed(
        self,
        user_history: List[Tuple[str, float]],
        top_k: int = 20,
    ) -> List[ScoredResult]:
        user_emb = self._encode_user_from_history(user_history)
        excluded = self._get_excluded_ids(user_history)
        fetch = top_k + len(excluded)

        results = self._qdrant_client.query_points(
            collection_name=self.cfg.qdrant.collection_name,
            query=user_emb.tolist(),
            limit=fetch,
        ).points

        scored = []
        for r in results:
            biz_id = r.payload.get("business_id", "")
            if biz_id in excluded:
                continue
            scored.append(ScoredResult(
                business_id=biz_id,
                name=r.payload.get("name", ""),
                score=r.score,
                filter_match=1.0,
                payload=r.payload,
            ))
        return scored[:top_k]

    def cold_start_search(
        self,
        filters: FilterSpec,
        top_k: int = 20,
    ) -> List[ScoredResult]:
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
        if filters.price_range_max is not None:
            conditions.append(FieldCondition(key="price_range", range=Range(lte=filters.price_range_max)))

        q_filter = Filter(must=conditions) if conditions else None

        scroll_results, _ = self._qdrant_client.scroll(
            collection_name=self.cfg.qdrant.collection_name,
            scroll_filter=q_filter,
            limit=top_k * 5,
            with_vectors=False,
            with_payload=True,
        )

        scored = []
        for point in scroll_results:
            p = point.payload or {}
            popularity = p.get("stars", 0) * (p.get("review_count", 0) ** 0.5)
            scored.append(ScoredResult(
                business_id=p.get("business_id", ""),
                name=p.get("name", ""),
                score=round(popularity / 100, 4),
                filter_match=1.0,
                payload=p,
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    @torch.no_grad()
    def _encode_user_from_history(
        self,
        history: List[Tuple[str, float]],
    ) -> np.ndarray:
        max_s = self.cfg.model.user_tower.max_history
        valid = [(self._biz_id_to_idx[bid], r) for bid, r in history if bid in self._biz_id_to_idx]
        valid = valid[-max_s:]

        if not valid:
            return np.zeros(self.cfg.model.restaurant_tower.output_dim, dtype=np.float32)

        hist_items = torch.zeros(1, max_s, dtype=torch.long, device=self.device)
        hist_ratings = torch.zeros(1, max_s, dtype=torch.float, device=self.device)
        hist_mask = torch.zeros(1, max_s, dtype=torch.bool, device=self.device)

        for j, (item_idx, rating) in enumerate(valid):
            hist_items[0, j] = item_idx
            hist_ratings[0, j] = rating
            hist_mask[0, j] = True

        user_emb = self._model.encode_user(hist_items, hist_ratings, hist_mask)
        return user_emb.squeeze(0).cpu().numpy()
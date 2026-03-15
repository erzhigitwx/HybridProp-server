from __future__ import annotations

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from ml.config import QdrantConfig

logger = logging.getLogger(__name__)


class QdrantIndexer:
    def __init__(self, cfg: QdrantConfig):
        self.cfg = cfg
        self._connect()

    def _connect(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(host=self.cfg.host, port=self.cfg.port)

        dist_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        self.distance = dist_map.get(self.cfg.distance, Distance.COSINE)
        logger.info("Connected to Qdrant at %s:%d", self.cfg.host, self.cfg.port)

    def create_collection(self, recreate: bool = True):
        from qdrant_client.models import VectorParams

        if recreate:
            try:
                self.client.delete_collection(self.cfg.collection_name)
                logger.info("Deleted existing collection: %s", self.cfg.collection_name)
            except Exception:
                pass

        self.client.create_collection(
            collection_name=self.cfg.collection_name,
            vectors_config=VectorParams(
                size=self.cfg.vector_dim,
                distance=self.distance,
            ),
        )
        logger.info(
            "Created collection '%s' (dim=%d, dist=%s)",
            self.cfg.collection_name, self.cfg.vector_dim, self.cfg.distance,
        )

    def create_payload_indices(self):
        from qdrant_client.models import PayloadSchemaType

        indices = {
            "city": PayloadSchemaType.KEYWORD,
            "state": PayloadSchemaType.KEYWORD,
            "stars": PayloadSchemaType.FLOAT,
            "price_range": PayloadSchemaType.INTEGER,
            "review_count": PayloadSchemaType.INTEGER,
            "is_open": PayloadSchemaType.BOOL,
        }
        for field_name, schema_type in indices.items():
            self.client.create_payload_index(
                collection_name=self.cfg.collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
        logger.info("Created %d payload indices", len(indices))

    def upsert_restaurants(
        self,
        embeddings: np.ndarray,
        business_ids: List[str],
        businesses_df: pd.DataFrame,
    ):
        from qdrant_client.models import PointStruct

        biz_lookup = businesses_df.set_index("business_id").to_dict("index")

        points = []
        for idx, (biz_id, emb) in enumerate(zip(business_ids, embeddings)):
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
                "address": meta.get("address", ""),
            }
            points.append(PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload=payload,
            ))

            if len(points) >= self.cfg.batch_size:
                self.client.upsert(
                    collection_name=self.cfg.collection_name,
                    points=points,
                )
                points = []

        if points:
            self.client.upsert(
                collection_name=self.cfg.collection_name,
                points=points,
            )

        logger.info("Upserted %d restaurants to Qdrant", len(business_ids))

    def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.cfg.collection_name)
        return {
            "points_count": info.points_count,
            "status": str(info.status),
        }
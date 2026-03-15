from __future__ import annotations

import logging
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from ml.config import TextEmbConfig

logger = logging.getLogger(__name__)


class TextEncoder:
    def __init__(self, cfg: TextEmbConfig, device_str: str = "cuda"):
        self.cfg = cfg
        self.device_str = device_str
        self._load_model()

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading text model: %s", self.cfg.model_name)
        self.model = SentenceTransformer(
            self.cfg.model_name,
            device=self.device_str,
        )
        self.model.max_seq_length = self.cfg.max_seq_length

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        logger.info("Encoding %d texts...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def encode_restaurant_texts(
        self,
        review_texts_df,
    ) -> Dict[str, np.ndarray]:
        biz_ids = review_texts_df["business_id"].tolist()
        texts = review_texts_df["combined_text"].tolist()
        texts = [t[: self.cfg.max_seq_length * 4] if isinstance(t, str) else "" for t in texts]

        embeddings = self.encode_texts(texts)

        result = {}
        for biz_id, emb in zip(biz_ids, embeddings):
            result[biz_id] = emb

        logger.info("Encoded text embeddings for %d restaurants", len(result))
        return result

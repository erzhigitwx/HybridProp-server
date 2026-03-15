from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ml.config import CLIPConfig

logger = logging.getLogger(__name__)


class CLIPEncoder:
    def __init__(self, cfg: CLIPConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._load_model()

    def _load_model(self):
        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP model: %s", self.cfg.model_name)
        self.model = CLIPModel.from_pretrained(self.cfg.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.cfg.model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        all_embs = []
        bs = self.cfg.batch_size

        for i in tqdm(range(0, len(image_paths), bs), desc="CLIP encoding"):
            batch_paths = image_paths[i : i + bs]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", p, e)
                    images.append(Image.new("RGB", (self.cfg.image_size, self.cfg.image_size)))

            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embs = self.model.get_image_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().numpy())

        return np.concatenate(all_embs, axis=0)

    def aggregate_per_restaurant(
        self,
        photo_df,
        embeddings: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        weights_map = self.cfg.photo_weights
        result: Dict[str, np.ndarray] = {}

        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, row in photo_df.iterrows():
            groups[row["business_id"]].append(idx)

        for biz_id, indices in groups.items():
            vecs = []
            weights = []
            for i in indices:
                label = photo_df.iloc[i].get("label", "inside")
                w = weights_map.get(label, 1.0)
                vecs.append(embeddings[i])
                weights.append(w)

            vecs = np.stack(vecs)
            weights = np.array(weights)
            weights /= weights.sum()
            agg = (vecs * weights[:, None]).sum(axis=0)
            agg /= np.linalg.norm(agg) + 1e-8
            result[biz_id] = agg

        logger.info("Aggregated CLIP embeddings for %d restaurants", len(result))
        return result

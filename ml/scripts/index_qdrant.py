import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)

from ml.config import load_config
from ml.embeddings.aggregator import load_embeddings
from ml.models.two_tower import TwoTowerModel
from ml.training.checkpoint import CheckpointManager
from ml.inference.qdrant_indexer import QdrantIndexer
from ml.utils.device import get_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Index embeddings to Qdrant")
    parser.add_argument("--config", type=str, default="ml/config/config.yaml")
    parser.add_argument("--no-recreate", action="store_true", help="Don't recreate collection")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    features, business_ids = load_embeddings(cfg.paths.embeddings)
    model = TwoTowerModel(cfg.model, features)

    ckpt_mgr = CheckpointManager(cfg.paths.checkpoints)
    ckpt = ckpt_mgr.load_best(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logging.info("Loaded model from epoch %d", ckpt["epoch"])

    import torch
    with torch.no_grad():
        item_embs = model.encode_all_items(batch_size=1024).numpy()
    logging.info("Encoded %d restaurant embeddings (dim=%d)", item_embs.shape[0], item_embs.shape[1])

    businesses = pd.read_parquet(f"{cfg.paths.processed}/businesses.parquet")

    indexer = QdrantIndexer(cfg.qdrant)
    indexer.create_collection(recreate=not args.no_recreate)
    indexer.create_payload_indices()
    indexer.upsert_restaurants(item_embs, business_ids, businesses)

    info = indexer.get_collection_info()
    print(f"\n=== Qdrant Collection ===")
    print(f"  Points:  {info['points_count']:>8,}")
    print(f"  Status:  {info['status']}")


if __name__ == "__main__":
    main()

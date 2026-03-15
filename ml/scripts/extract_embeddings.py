import argparse
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)

from ml.config import load_config
from ml.embeddings.clip_encoder import CLIPEncoder
from ml.embeddings.text_encoder import TextEncoder
from ml.embeddings.aggregator import (
    build_tabular_features,
    combine_embeddings,
    save_embeddings,
)
from ml.utils.device import get_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings")
    parser.add_argument("--config", type=str, default="ml/config/config.yaml")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP (use cached)")
    parser.add_argument("--skip-text", action="store_true", help="Skip text encoder (use cached)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    device_str = str(device)

    businesses = pd.read_parquet(f"{cfg.paths.processed}/businesses.parquet")
    photos = pd.read_parquet(f"{cfg.paths.processed}/photos.parquet")
    review_texts = pd.read_parquet(f"{cfg.paths.processed}/review_texts.parquet")
    biz_ids = businesses["business_id"].tolist()

    import numpy as np
    from pathlib import Path

    clip_cache = Path(cfg.paths.embeddings) / "clip_raw.npz"
    if args.skip_clip and clip_cache.exists():
        logging.info("Loading cached CLIP embeddings from %s", clip_cache)
        data = np.load(clip_cache, allow_pickle=True)
        clip_embs = dict(zip(data["ids"], data["embs"]))
    else:
        encoder = CLIPEncoder(cfg.embeddings.clip, device)
        image_paths = photos["image_path"].tolist()
        raw_embs = encoder.encode_images(image_paths)
        clip_embs = encoder.aggregate_per_restaurant(photos, raw_embs)
        Path(cfg.paths.embeddings).mkdir(parents=True, exist_ok=True)
        np.savez(
            clip_cache,
            ids=list(clip_embs.keys()),
            embs=list(clip_embs.values()),
        )

    text_cache = Path(cfg.paths.embeddings) / "text_raw.npz"
    if args.skip_text and text_cache.exists():
        logging.info("Loading cached text embeddings from %s", text_cache)
        data = np.load(text_cache, allow_pickle=True)
        text_embs = dict(zip(data["ids"], data["embs"]))
    else:
        text_enc = TextEncoder(cfg.embeddings.text, device_str)
        text_embs = text_enc.encode_restaurant_texts(review_texts)
        np.savez(
            text_cache,
            ids=list(text_embs.keys()),
            embs=list(text_embs.values()),
        )

    tabular_embs, tab_dim = build_tabular_features(businesses)

    combined, valid_ids = combine_embeddings(
        business_ids=biz_ids,
        clip_embs=clip_embs,
        text_embs=text_embs,
        tabular_embs=tabular_embs,
        clip_dim=cfg.embeddings.clip.dim,
        text_dim=cfg.embeddings.text.dim,
    )

    save_embeddings(combined, valid_ids, cfg.paths.embeddings)

    print(f"\n=== Embeddings Summary ===")
    print(f"  Restaurants with CLIP:  {len(clip_embs):>6,}")
    print(f"  Restaurants with text:  {len(text_embs):>6,}")
    print(f"  Combined matrix shape:  {combined.shape}")


if __name__ == "__main__":
    main()

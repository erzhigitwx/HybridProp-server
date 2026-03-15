import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Slim dataset for deployment")
    parser.add_argument("--config", type=str, default="ml/config/config.yaml")
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--max-reviews", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from ml.config import load_config
    cfg = load_config(args.config)

    processed = Path(cfg.paths.processed)
    raw_dir = Path(cfg.paths.raw_data)
    photos_dir = Path(cfg.paths.photos_dir)
    photos_json_path = Path(cfg.paths.photos_json)
    embeddings_dir = Path(cfg.paths.embeddings)

    businesses = pd.read_parquet(processed / "businesses.parquet")
    log.info("Total restaurants: %d", len(businesses))

    businesses["popularity"] = businesses["stars"] * (businesses["review_count"] ** 0.5)
    top = businesses.nlargest(args.top_n, "popularity")
    keep_biz_ids = set(top["business_id"].unique())
    log.info("Keeping top %d restaurants", len(keep_biz_ids))

    top_cities = top["city"].value_counts().head(10)
    log.info("Top cities in kept set:\n%s", top_cities.to_string())

    reviews = pd.read_parquet(processed / "reviews.parquet")
    reviews_kept = reviews[reviews["business_id"].isin(keep_biz_ids)]
    reviews_kept = (
        reviews_kept
        .sort_values("useful", ascending=False)
        .groupby("business_id")
        .head(args.max_reviews)
    )
    keep_user_ids = set(reviews_kept["user_id"].unique())
    log.info("Reviews: %d → %d", len(reviews), len(reviews_kept))
    log.info("Users: kept %d", len(keep_user_ids))

    photos = pd.read_parquet(processed / "photos.parquet")
    photos_kept = photos[photos["business_id"].isin(keep_biz_ids)]
    keep_photo_ids = set(photos_kept["photo_id"].unique())
    log.info("Photos: %d → %d", len(photos), len(photos_kept))

    emb_ids_path = embeddings_dir / "restaurant_ids.csv"
    emb_matrix_path = embeddings_dir / "restaurant_features.npy"
    if emb_ids_path.exists() and emb_matrix_path.exists():
        all_ids = pd.read_csv(emb_ids_path, header=None)[0].tolist()
        all_embs = np.load(emb_matrix_path)
        mask = [i for i, bid in enumerate(all_ids) if bid in keep_biz_ids]
        new_ids = [all_ids[i] for i in mask]
        new_embs = all_embs[mask]
        log.info("Embeddings: %d → %d", len(all_ids), len(new_ids))
    else:
        new_ids, new_embs = None, None
        log.warning("Embeddings not found, skipping")

    sizes = {}
    sizes["businesses"] = len(top)
    sizes["reviews"] = len(reviews_kept)
    sizes["photos_parquet"] = len(photos_kept)
    sizes["photo_files"] = len(keep_photo_ids)
    sizes["users"] = len(keep_user_ids)
    if new_ids:
        sizes["embeddings"] = len(new_ids)

    log.info("\n=== Summary ===")
    for k, v in sizes.items():
        log.info("  %-20s %d", k, v)

    if args.dry_run:
        log.info("\nDRY RUN — no changes made")
        return

    log.info("\nWriting slimmed processed files...")
    top.to_parquet(processed / "businesses.parquet", index=False)
    reviews_kept.to_parquet(processed / "reviews.parquet", index=False)
    photos_kept.to_parquet(processed / "photos.parquet", index=False)

    users = pd.read_parquet(processed / "users.parquet")
    users_kept = users[users["user_id"].isin(keep_user_ids)]
    users_kept.to_parquet(processed / "users.parquet", index=False)

    if (processed / "tips.parquet").exists():
        tips = pd.read_parquet(processed / "tips.parquet")
        tips_kept = tips[tips["business_id"].isin(keep_biz_ids)]
        tips_kept.to_parquet(processed / "tips.parquet", index=False)

    if (processed / "review_texts.parquet").exists():
        rt = pd.read_parquet(processed / "review_texts.parquet")
        rt_kept = rt[rt["business_id"].isin(keep_biz_ids)]
        rt_kept.to_parquet(processed / "review_texts.parquet", index=False)

    if (processed / "user_history.parquet").exists():
        uh = pd.read_parquet(processed / "user_history.parquet")
        uh_kept = uh[uh["business_id"].isin(keep_biz_ids) & uh["user_id"].isin(keep_user_ids)]
        uh_kept.to_parquet(processed / "user_history.parquet", index=False)

    if new_ids is not None:
        np.save(emb_matrix_path, new_embs)
        pd.Series(new_ids).to_csv(emb_ids_path, index=False, header=False)
        log.info("Embeddings slimmed: %d vectors", len(new_ids))

    log.info("Trimming raw JSON files...")
    raw_files = [
        ("yelp_academic_dataset_business.json", "business_id", keep_biz_ids),
        ("yelp_academic_dataset_review.json", "business_id", keep_biz_ids),
        ("yelp_academic_dataset_user.json", "user_id", keep_user_ids),
        ("yelp_academic_dataset_tip.json", "business_id", keep_biz_ids),
        ("yelp_academic_dataset_checkin.json", "business_id", keep_biz_ids),
    ]
    for filename, key, valid_ids in raw_files:
        src = raw_dir / filename
        if not src.exists():
            continue
        kept_lines = []
        with open(src, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get(key) in valid_ids:
                    kept_lines.append(json.dumps(obj, ensure_ascii=False))
        with open(src, "w", encoding="utf-8") as fout:
            fout.write("\n".join(kept_lines) + "\n")
        log.info("  %s → %d rows", filename, len(kept_lines))

    if photos_json_path.exists():
        kept_lines = []
        with open(photos_json_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("photo_id") in keep_photo_ids:
                    kept_lines.append(json.dumps(obj, ensure_ascii=False))
        with open(photos_json_path, "w", encoding="utf-8") as fout:
            fout.write("\n".join(kept_lines) + "\n")
        log.info("  photos.json → %d rows", len(kept_lines))

    if photos_dir.exists():
        deleted = 0
        for f in photos_dir.glob("*.jpg"):
            if f.stem not in keep_photo_ids:
                f.unlink()
                deleted += 1
        log.info("  Deleted %d unused photo files", deleted)

    total_mb = 0
    for d in [raw_dir, photos_dir.parent, processed, embeddings_dir]:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    total_mb += f.stat().st_size
    log.info("\n=== Final dataset size: %.0f MB ===", total_mb / 1e6)


if __name__ == "__main__":
    main()
import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_valid_ids(processed_dir: str):
    biz = pd.read_parquet(Path(processed_dir) / "businesses.parquet")
    biz_ids = set(biz["business_id"].unique())
    log.info("Valid restaurants: %d", len(biz_ids))

    reviews = pd.read_parquet(Path(processed_dir) / "reviews.parquet")
    user_ids = set(reviews["user_id"].unique())
    review_ids = set(reviews["review_id"].unique()) if "review_id" in reviews.columns else set()
    log.info("Valid users: %d", len(user_ids))

    photos = pd.read_parquet(Path(processed_dir) / "photos.parquet")
    photo_ids = set(photos["photo_id"].unique())
    log.info("Valid photos: %d", len(photo_ids))

    return biz_ids, user_ids, photo_ids


def filter_jsonl(input_path: Path, output_path: Path, key: str, valid_ids: set):
    if not input_path.exists():
        log.warning("File not found: %s", input_path)
        return 0, 0

    total = 0
    kept = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get(key) in valid_ids:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    log.info("  %s: %d → %d (removed %d)", input_path.name, total, kept, total - kept)
    return total, kept


def trim_photos(photos_dir: Path, valid_photo_ids: set, dry_run: bool):
    if not photos_dir.exists():
        log.warning("Photos dir not found: %s", photos_dir)
        return

    all_files = list(photos_dir.glob("*.jpg"))
    to_delete = []
    for f in all_files:
        photo_id = f.stem
        if photo_id not in valid_photo_ids:
            to_delete.append(f)

    log.info("Photos: %d total, %d to keep, %d to delete", len(all_files), len(all_files) - len(to_delete), len(to_delete))

    if dry_run:
        log.info("  DRY RUN — no files deleted")
        return

    for f in to_delete:
        f.unlink()
    log.info("  Deleted %d unused photos", len(to_delete))


def main():
    parser = argparse.ArgumentParser(description="Trim Yelp dataset to project data only")
    parser.add_argument("--config", type=str, default="ml/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    args = parser.parse_args()

    from ml.config import load_config
    cfg = load_config(args.config)

    raw_dir = Path(cfg.paths.raw_data)
    photos_dir = Path(cfg.paths.photos_dir)
    photos_json = Path(cfg.paths.photos_json)
    processed_dir = Path(cfg.paths.processed)

    biz_ids, user_ids, photo_ids = load_valid_ids(str(processed_dir))

    files_to_filter = [
        ("yelp_academic_dataset_business.json", "business_id", biz_ids),
        ("yelp_academic_dataset_review.json", "business_id", biz_ids),
        ("yelp_academic_dataset_user.json", "user_id", user_ids),
        ("yelp_academic_dataset_tip.json", "business_id", biz_ids),
        ("yelp_academic_dataset_checkin.json", "business_id", biz_ids),
    ]

    log.info("\n=== Filtering JSON files ===")
    for filename, key, valid in files_to_filter:
        src = raw_dir / filename
        if not src.exists():
            log.warning("  Skipping %s (not found)", filename)
            continue

        tmp = src.with_suffix(".tmp")
        if args.dry_run:
            total = sum(1 for _ in open(src, encoding="utf-8"))
            log.info("  %s: %d lines (dry run, no changes)", filename, total)
        else:
            filter_jsonl(src, tmp, key, valid)
            src.unlink()
            tmp.rename(src)

    if photos_json.exists():
        log.info("\n=== Filtering photos.json ===")
        if args.dry_run:
            total = sum(1 for _ in open(photos_json, encoding="utf-8"))
            log.info("  photos.json: %d lines (dry run)", total)
        else:
            tmp = photos_json.with_suffix(".tmp")
            filter_jsonl(photos_json, tmp, "photo_id", photo_ids)
            photos_json.unlink()
            tmp.rename(photos_json)

    log.info("\n=== Trimming photo files ===")
    trim_photos(photos_dir, photo_ids, args.dry_run)

    if not args.dry_run:
        total_size = 0
        for d in [raw_dir, photos_dir.parent]:
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
        log.info("\n=== Done! Dataset size: %.1f GB ===", total_size / 1e9)


if __name__ == "__main__":
    main()
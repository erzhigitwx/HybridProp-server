from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from ml.config import Config

logger = logging.getLogger(__name__)


def _read_jsonl(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

_BOOL_MAP = {"True": 1, "False": 0, "None": np.nan}


def _parse_bool_attr(val) -> float:
    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip().strip("'\"")
    return _BOOL_MAP.get(s, np.nan)


def _parse_price_range(val) -> float:
    if val is None or pd.isna(val):
        return np.nan
    try:
        return float(str(val).strip().strip("'\""))
    except ValueError:
        return np.nan

class DataPreprocessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.raw = Path(cfg.paths.raw_data)
        self.out = Path(cfg.paths.processed)
        self.out.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading raw data...")
        businesses = self._load_businesses()
        photos = self._load_photos(businesses.business_id.unique())
        reviews = self._load_reviews(businesses.business_id.unique())
        users = self._load_users(reviews.user_id.unique())
        tips = self._load_tips(businesses.business_id.unique())

        logger.info("Engineering features...")
        businesses = self._engineer_business_features(businesses)
        review_texts = self._aggregate_review_texts(reviews)
        user_history = self._build_user_history(reviews)

        businesses.to_parquet(self.out / "businesses.parquet", index=False)
        photos.to_parquet(self.out / "photos.parquet", index=False)
        reviews.to_parquet(self.out / "reviews.parquet", index=False)
        users.to_parquet(self.out / "users.parquet", index=False)
        tips.to_parquet(self.out / "tips.parquet", index=False)
        review_texts.to_parquet(self.out / "review_texts.parquet", index=False)
        user_history.to_parquet(self.out / "user_history.parquet", index=False)

        logger.info(
            "Preprocessing complete: %d restaurants, %d photos, %d reviews, %d users",
            len(businesses), len(photos), len(reviews), len(users),
        )
        return {
            "businesses": businesses,
            "photos": photos,
            "reviews": reviews,
            "users": users,
            "tips": tips,
            "review_texts": review_texts,
            "user_history": user_history,
        }

    def _load_businesses(self) -> pd.DataFrame:
        df = _read_jsonl(self.raw / "yelp_academic_dataset_business.json")
        cats = set(self.cfg.data.restaurant_categories)

        def _is_restaurant(cat_str):
            if not isinstance(cat_str, str):
                return False
            return bool(cats & {c.strip() for c in cat_str.split(",")})

        df = df[df["categories"].apply(_is_restaurant)].copy()
        df = df[df["is_open"] == 1]
        df = df[df["review_count"] >= self.cfg.data.min_reviews]
        df = df.reset_index(drop=True)
        logger.info("Filtered to %d open restaurants with >=%d reviews", len(df), self.cfg.data.min_reviews)
        return df

    def _load_photos(self, biz_ids: Set[str]) -> pd.DataFrame:
        path = Path(self.cfg.paths.photos_json)
        df = _read_jsonl(path)
        df = df[df["business_id"].isin(biz_ids)].copy()
        photos_dir = Path(self.cfg.paths.photos_dir)
        df["image_path"] = df["photo_id"].apply(lambda pid: str(photos_dir / f"{pid}.jpg"))
        df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
        missing = (~df["exists"]).sum()
        if missing > 0:
            logger.warning("%d photo files not found on disk, skipping them", missing)
        df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
        logger.info("Loaded %d photos for %d restaurants", len(df), df["business_id"].nunique())
        return df

    def _load_reviews(self, biz_ids: Set[str]) -> pd.DataFrame:
        df = _read_jsonl(self.raw / "yelp_academic_dataset_review.json")
        df = df[df["business_id"].isin(biz_ids)].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("Loaded %d reviews", len(df))
        return df

    def _load_users(self, user_ids: Set[str]) -> pd.DataFrame:
        df = _read_jsonl(self.raw / "yelp_academic_dataset_user.json")
        df = df[df["user_id"].isin(user_ids)].copy()
        df = df[["user_id", "name", "review_count", "yelping_since",
                  "useful", "funny", "cool", "fans", "average_stars", "elite"]].copy()
        df["yelping_since"] = pd.to_datetime(df["yelping_since"])
        logger.info("Loaded %d users", len(df))
        return df

    def _load_tips(self, biz_ids: Set[str]) -> pd.DataFrame:
        df = _read_jsonl(self.raw / "yelp_academic_dataset_tip.json")
        df = df[df["business_id"].isin(biz_ids)].copy()
        df["date"] = pd.to_datetime(df["date"])
        logger.info("Loaded %d tips", len(df))
        return df

    def _engineer_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        attrs = df["attributes"].apply(lambda x: x if isinstance(x, dict) else {})
        df["price_range"] = attrs.apply(lambda a: _parse_price_range(a.get("RestaurantsPriceRange2")))
        df["wifi"] = attrs.apply(lambda a: 1.0 if "free" in str(a.get("WiFi", "")).lower() else 0.0)
        df["outdoor_seating"] = attrs.apply(lambda a: _parse_bool_attr(a.get("OutdoorSeating")))
        df["good_for_kids"] = attrs.apply(lambda a: _parse_bool_attr(a.get("GoodForKids")))
        df["delivery"] = attrs.apply(lambda a: _parse_bool_attr(a.get("RestaurantsDelivery")))
        df["takeout"] = attrs.apply(lambda a: _parse_bool_attr(a.get("RestaurantsTakeOut")))
        df["reservations"] = attrs.apply(lambda a: _parse_bool_attr(a.get("RestaurantsReservations")))
        df["wheelchair"] = attrs.apply(lambda a: _parse_bool_attr(a.get("WheelchairAccessible")))
        df["bike_parking"] = attrs.apply(lambda a: _parse_bool_attr(a.get("BikeParking")))

        le_city = LabelEncoder()
        df["city_encoded"] = le_city.fit_transform(df["city"].fillna("unknown"))

        le_state = LabelEncoder()
        df["state_encoded"] = le_state.fit_transform(df["state"].fillna("unknown"))

        df["category_list"] = df["categories"].apply(
            lambda c: [x.strip() for x in c.split(",")] if isinstance(c, str) else []
        )
        mlb = MultiLabelBinarizer()
        cat_matrix = mlb.fit_transform(df["category_list"])
        freq = cat_matrix.sum(axis=0)
        top_idx = np.argsort(freq)[-50:]
        cat_cols = [f"cat_{mlb.classes_[i]}" for i in top_idx]
        for i, col in zip(top_idx, cat_cols):
            df[col] = cat_matrix[:, i]

        num_cols = ["stars", "review_count", "price_range", "wifi", "outdoor_seating",
                    "good_for_kids", "delivery", "takeout", "reservations",
                    "wheelchair", "bike_parking"]
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def _aggregate_review_texts(self, reviews: pd.DataFrame) -> pd.DataFrame:
        max_n = self.cfg.data.max_reviews_per_business
        top = (
            reviews.sort_values("useful", ascending=False)
            .groupby("business_id")
            .head(max_n)
        )
        agg = (
            top.groupby("business_id")["text"]
            .apply(lambda texts: " ".join(texts.astype(str)))
            .reset_index()
            .rename(columns={"text": "combined_text"})
        )
        stats = (
            reviews.groupby("business_id")
            .agg(avg_review_stars=("stars", "mean"), total_reviews=("stars", "count"))
            .reset_index()
        )
        agg = agg.merge(stats, on="business_id", how="left")
        return agg

    def _build_user_history(self, reviews: pd.DataFrame) -> pd.DataFrame:
        min_inter = self.cfg.data.min_user_interactions
        counts = reviews.groupby("user_id").size()
        valid_users = counts[counts >= min_inter].index

        history = (
            reviews[reviews["user_id"].isin(valid_users)]
            [["user_id", "business_id", "stars", "date"]]
            .sort_values(["user_id", "date"])
            .reset_index(drop=True)
        )
        logger.info("Built history for %d users (min %d interactions)", history["user_id"].nunique(), min_inter)
        return history

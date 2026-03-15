from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from api.dependencies import AppState, get_state
from ml.config import Config

router = APIRouter(prefix="/restaurant", tags=["restaurant"])


class PhotoItem(BaseModel):
    photo_id: str
    label: str
    caption: str


class ReviewItem(BaseModel):
    text: str
    stars: float
    date: str
    useful: int


class RestaurantDetail(BaseModel):
    business_id: str
    name: str
    address: str
    city: str
    state: str
    stars: float
    review_count: int
    price_range: int
    categories: str
    latitude: float
    longitude: float
    photos: List[PhotoItem]
    reviews: List[ReviewItem]
    attributes: dict


@router.get("/{business_id}", response_model=RestaurantDetail)
async def get_restaurant(
    business_id: str,
    state: AppState = Depends(get_state),
):
    cfg = state.config
    processed = Path(cfg.paths.processed)

    businesses = pd.read_parquet(processed / "businesses.parquet")
    biz = businesses[businesses["business_id"] == business_id]
    if biz.empty:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    biz = biz.iloc[0]

    photos_df = pd.read_parquet(processed / "photos.parquet")
    photos = photos_df[photos_df["business_id"] == business_id]
    photo_items = [
        PhotoItem(
            photo_id=row["photo_id"],
            label=row.get("label", ""),
            caption=row.get("caption", ""),
        )
        for _, row in photos.iterrows()
    ]

    reviews_df = pd.read_parquet(processed / "reviews.parquet")
    revs = (
        reviews_df[reviews_df["business_id"] == business_id]
        .sort_values("useful", ascending=False)
        .head(20)
    )
    review_items = [
        ReviewItem(
            text=row["text"][:500],
            stars=row["stars"],
            date=str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10],
            useful=int(row["useful"]),
        )
        for _, row in revs.iterrows()
    ]

    attrs = {}
    if isinstance(biz.get("attributes"), dict):
        attrs = biz["attributes"]

    return RestaurantDetail(
        business_id=business_id,
        name=biz.get("name", ""),
        address=biz.get("address", ""),
        city=biz.get("city", ""),
        state=biz.get("state", ""),
        stars=float(biz.get("stars", 0)),
        review_count=int(biz.get("review_count", 0)),
        price_range=int(biz.get("price_range", 0)) if pd.notna(biz.get("price_range")) else 0,
        categories=biz.get("categories", ""),
        latitude=float(biz.get("latitude", 0)),
        longitude=float(biz.get("longitude", 0)),
        photos=photo_items,
        reviews=review_items,
        attributes=attrs,
    )
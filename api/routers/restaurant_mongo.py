from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_state, AppState

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
    db = state.mongo_db
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    biz = db.businesses.find_one({"business_id": business_id}, {"_id": 0})
    if not biz:
        raise HTTPException(status_code=404, detail="Restaurant not found")

    photos_cursor = db.photos.find({"business_id": business_id}, {"_id": 0})
    photo_items = [
        PhotoItem(
            photo_id=p.get("photo_id", ""),
            label=p.get("label", ""),
            caption=p.get("caption", ""),
        )
        for p in photos_cursor
    ]

    reviews_cursor = (
        db.reviews.find({"business_id": business_id}, {"_id": 0})
        .sort("useful", -1)
        .limit(20)
    )
    review_items = [
        ReviewItem(
            text=r.get("text", ""),
            stars=r.get("stars", 0),
            date=r.get("date", ""),
            useful=r.get("useful", 0),
        )
        for r in reviews_cursor
    ]

    return RestaurantDetail(
        business_id=business_id,
        name=biz.get("name", ""),
        address=biz.get("address", ""),
        city=biz.get("city", ""),
        state=biz.get("state", ""),
        stars=float(biz.get("stars", 0)),
        review_count=int(biz.get("review_count", 0)),
        price_range=int(biz.get("price_range", 0)),
        categories=biz.get("categories", ""),
        latitude=float(biz.get("latitude", 0)),
        longitude=float(biz.get("longitude", 0)),
        photos=photo_items,
        reviews=review_items,
        attributes=biz.get("attributes", {}),
    )
from __future__ import annotations
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from api.dependencies import AppState, get_state, get_recommender
from ml.inference.soft_filter import FilterSpec
from ml.inference.recommender import Recommender

router = APIRouter(prefix="/recommend", tags=["recommendations"])


class FilterRequest(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    stars_min: Optional[float] = Field(None, ge=1.0, le=5.0)
    stars_max: Optional[float] = Field(None, ge=1.0, le=5.0)
    price_range_min: Optional[int] = Field(None, ge=1, le=4)
    price_range_max: Optional[int] = Field(None, ge=1, le=4)
    categories: Optional[List[str]] = None


class SoftFilterRequest(BaseModel):
    user_id: str
    filters: FilterRequest
    top_k: int = Field(20, ge=1, le=100)
    softness: Optional[float] = Field(None, ge=0.0, le=1.0)


class SimilarRequest(BaseModel):
    business_id: str
    top_k: int = Field(10, ge=1, le=50)


class RestaurantResult(BaseModel):
    business_id: str
    name: str
    score: float
    filter_match: float
    city: str = ""
    state: str = ""
    stars: float = 0.0
    price_range: int = 0
    categories: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    soft_reasons: List[str] = []


class RecommendationResponse(BaseModel):
    results: List[RestaurantResult]
    total: int
    strict_count: int  # how many matched strict filters
    soft_count: int    # how many are soft matches

@router.post("/soft-filter", response_model=RecommendationResponse)
async def soft_filter_recommend(
    request: SoftFilterRequest,
    state: AppState = Depends(get_state),
    recommender: Recommender = Depends(get_recommender),
):
    user_history = state.get_user_history(request.user_id)

    filters = FilterSpec(
        city=request.filters.city,
        state=request.filters.state,
        stars_min=request.filters.stars_min,
        stars_max=request.filters.stars_max,
        price_range_min=request.filters.price_range_min,
        price_range_max=request.filters.price_range_max,
        categories=request.filters.categories,
    )

    if not user_history:
        results = recommender.cold_start_search(
            filters=filters,
            top_k=request.top_k,
        )
    else:
        results = recommender.recommend_with_soft_filter(
            user_history=user_history,
            filters=filters,
            top_k=request.top_k,
            softness=request.softness,
        )

    strict_count = sum(1 for r in results if r.filter_match >= 1.0)
    soft_count = sum(1 for r in results if r.filter_match < 1.0)

    return RecommendationResponse(
        results=[
            RestaurantResult(
                business_id=r.business_id,
                name=r.name,
                score=round(r.score, 4),
                filter_match=r.filter_match,
                city=r.payload.get("city", ""),
                state=r.payload.get("state", ""),
                stars=r.payload.get("stars", 0),
                price_range=r.payload.get("price_range", 0),
                categories=r.payload.get("categories", ""),
                latitude=r.payload.get("latitude", 0),
                longitude=r.payload.get("longitude", 0),
                soft_reasons=r.soft_reasons,
            )
            for r in results
        ],
        total=len(results),
        strict_count=strict_count,
        soft_count=soft_count,
    )


@router.post("/similar", response_model=RecommendationResponse)
async def find_similar(
    request: SimilarRequest,
    recommender: Recommender = Depends(get_recommender),
):
    results = recommender.find_similar(
        business_id=request.business_id,
        top_k=request.top_k,
    )

    return RecommendationResponse(
        results=[
            RestaurantResult(
                business_id=r.business_id,
                name=r.name,
                score=round(r.score, 4),
                filter_match=r.filter_match,
                city=r.payload.get("city", ""),
                state=r.payload.get("state", ""),
                stars=r.payload.get("stars", 0),
                price_range=r.payload.get("price_range", 0),
                categories=r.payload.get("categories", ""),
                latitude=r.payload.get("latitude", 0),
                longitude=r.payload.get("longitude", 0),
            )
            for r in results
        ],
        total=len(results),
        strict_count=len(results),
        soft_count=0,
    )


@router.get("/feed/{user_id}", response_model=RecommendationResponse)
async def personalized_feed(
    user_id: str,
    top_k: int = 20,
    state: AppState = Depends(get_state),
    recommender: Recommender = Depends(get_recommender),
):
    user_history = state.get_user_history(user_id)

    if not user_history:
        results = recommender.cold_start_search(
            filters=FilterSpec(),
            top_k=top_k,
        )
    else:
        results = recommender.personalized_feed(
            user_history=user_history,
            top_k=top_k,
        )

    return RecommendationResponse(
        results=[
            RestaurantResult(
                business_id=r.business_id,
                name=r.name,
                score=round(r.score, 4),
                filter_match=1.0,
                city=r.payload.get("city", ""),
                state=r.payload.get("state", ""),
                stars=r.payload.get("stars", 0),
                price_range=r.payload.get("price_range", 0),
                categories=r.payload.get("categories", ""),
                latitude=r.payload.get("latitude", 0),
                longitude=r.payload.get("longitude", 0),
            )
            for r in results
        ],
        total=len(results),
        strict_count=len(results),
        soft_count=0,
    )
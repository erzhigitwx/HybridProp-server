from __future__ import annotations
from enum import Enum
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import AppState, get_state

router = APIRouter(prefix="/feedback", tags=["feedback"])


class InteractionType(str, Enum):
    like = "like"          # explicit positive → rating 5.0
    dislike = "dislike"    # explicit negative → rating 1.0
    view = "view"          # implicit positive → rating 3.5
    click = "click"        # implicit positive → rating 3.0
    skip = "skip"          # implicit negative → rating 2.0
    bookmark = "bookmark"  # strong positive → rating 4.5


INTERACTION_RATING = {
    InteractionType.like: 5.0,
    InteractionType.dislike: 1.0,
    InteractionType.view: 3.5,
    InteractionType.click: 3.0,
    InteractionType.skip: 2.0,
    InteractionType.bookmark: 4.5,
}


class InteractionRequest(BaseModel):
    user_id: str
    business_id: str
    interaction_type: InteractionType


class BatchInteractionRequest(BaseModel):
    interactions: List[InteractionRequest] = Field(..., min_length=1, max_length=100)


class HistoryEntry(BaseModel):
    business_id: str
    implicit_rating: float


class HistoryResponse(BaseModel):
    user_id: str
    interactions: List[HistoryEntry]
    total: int


class FeedbackResponse(BaseModel):
    status: str
    user_id: str
    history_size: int


@router.post("/interaction", response_model=FeedbackResponse)
async def record_interaction(
    request: InteractionRequest,
    state: AppState = Depends(get_state),
):
    rating = INTERACTION_RATING[request.interaction_type]
    state.add_interaction(request.user_id, request.business_id, rating)

    history = state.get_user_history(request.user_id)
    return FeedbackResponse(
        status="ok",
        user_id=request.user_id,
        history_size=len(history),
    )


@router.post("/batch", response_model=FeedbackResponse)
async def record_batch(
    request: BatchInteractionRequest,
    state: AppState = Depends(get_state),
):
    user_id = request.interactions[0].user_id
    for inter in request.interactions:
        if inter.user_id != user_id:
            raise HTTPException(
                status_code=400,
                detail="All interactions in a batch must belong to the same user",
            )
        rating = INTERACTION_RATING[inter.interaction_type]
        state.add_interaction(inter.user_id, inter.business_id, rating)

    history = state.get_user_history(user_id)
    return FeedbackResponse(
        status="ok",
        user_id=user_id,
        history_size=len(history),
    )


@router.get("/history/{user_id}", response_model=HistoryResponse)
async def get_history(
    user_id: str,
    state: AppState = Depends(get_state),
):
    history = state.get_user_history(user_id)
    return HistoryResponse(
        user_id=user_id,
        interactions=[
            HistoryEntry(business_id=bid, implicit_rating=rating)
            for bid, rating in history
        ],
        total=len(history),
    )


@router.delete("/history/{user_id}", response_model=FeedbackResponse)
async def clear_history(
    user_id: str,
    state: AppState = Depends(get_state),
):
    if user_id in state.user_sessions:
        del state.user_sessions[user_id]
    return FeedbackResponse(
        status="cleared",
        user_id=user_id,
        history_size=0,
    )

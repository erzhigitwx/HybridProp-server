from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import app_state
from api.routers.recommend import router as recommend_router
from api.routers.feedback import router as feedback_router
from api.routers.restaurant_mongo import router as restaurant_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    config_path = os.environ.get("HYBRIDPROP_CONFIG", "ml/config/config.yaml")
    logger.info("Initializing app with config: %s", config_path)
    app_state.initialize(config_path)
    logger.info("App ready!")
    yield
    # shutdown
    logger.info("Shutting down")


app = 4444444444444(
    title="HybridProp RecSys",
    description=(
        "Multimodal restaurant recommendation system with soft-boundary filtering. "
        "Combines CLIP visual embeddings, review text embeddings, and tabular features "
        "in a two-tower architecture with Qdrant vector search."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response


app.include_router(recommend_router)
app.include_router(feedback_router)
app.include_router(restaurant_router)

@app.get("/health", tags=["system"])
async def health():
    qdrant_ok = False
    try:
        info = app_state.recommender._qdrant_client.get_collection(
            app_state.config.qdrant.collection_name
        )
        qdrant_ok = info.status.name == "GREEN" or info.points_count > 0
    except Exception:
        pass

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "model_loaded": app_state.recommender is not None,
        "qdrant_connected": qdrant_ok,
        "active_sessions": len(app_state.user_sessions),
    }


@app.get("/", tags=["system"])
async def root():
    return {
        "service": "HybridProp RecSys",
        "version": "0.1.0",
        "docs": "/docs",
    }


from fastapi.staticfiles import StaticFiles
from pathlib import Path

photos_dir = Path(os.environ.get("PHOTOS_DIR", "ml/data/dataset/yelp_photos/photos"))
if photos_dir.exists():
    app.mount("/photos", StaticFiles(directory=str(photos_dir)), name="photos")
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ml.config import Config, load_config
from ml.inference.recommender import Recommender

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self):
        self.config: Optional[Config] = None
        self.recommender: Optional[Recommender] = None
        self.mongo_db = None
        self.user_sessions: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def initialize(self, config_path: str = "ml/config/config.yaml"):
        self.config = load_config(config_path)
        self.recommender = Recommender(self.config)
        self.recommender.load()
        self._connect_mongo()
        logger.info("AppState initialized")

    def _connect_mongo(self):
        import os
        uri = os.environ.get("MONGODB_URI")
        if uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                client.admin.command("ping")
                self.mongo_db = client["hybridprop"]
                logger.info("MongoDB connected")
            except Exception as e:
                logger.warning("MongoDB connection failed: %s (restaurant detail page will be unavailable)", e)
                self.mongo_db = None
        else:
            logger.info("MONGODB_URI not set, restaurant detail will use parquet fallback")

    def get_user_history(self, user_id: str) -> List[Tuple[str, float]]:
        return self.user_sessions.get(user_id, [])

    def add_interaction(self, user_id: str, business_id: str, rating: float):
        self.user_sessions[user_id].append((business_id, rating))
        max_hist = self.config.model.user_tower.max_history if self.config else 50
        self.user_sessions[user_id] = self.user_sessions[user_id][-max_hist:]


app_state = AppState()

def get_state() -> AppState:
    return app_state


def get_recommender() -> Recommender:
    if app_state.recommender is None:
        raise RuntimeError("Recommender not initialized. Call app_state.initialize() first.")
    return app_state.recommender
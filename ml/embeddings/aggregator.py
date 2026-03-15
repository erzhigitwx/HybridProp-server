from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TABULAR_COLS = [
    "stars", "review_count", "price_range",
    "wifi", "outdoor_seating", "good_for_kids",
    "delivery", "takeout", "reservations",
    "wheelchair", "bike_parking",
    "city_encoded", "state_encoded",
]


def build_tabular_features(businesses: pd.DataFrame) -> Dict[str, np.ndarray]:
    feature_cols = [c for c in TABULAR_COLS if c in businesses.columns]
    cat_cols = [c for c in businesses.columns if c.startswith("cat_")]
    all_cols = feature_cols + cat_cols

    matrix = businesses[all_cols].values.astype(np.float32)
    for i, col in enumerate(all_cols):
        if col not in cat_cols:
            mu = np.nanmean(matrix[:, i])
            std = np.nanstd(matrix[:, i]) + 1e-8
            matrix[:, i] = (matrix[:, i] - mu) / std

    matrix = np.nan_to_num(matrix, nan=0.0)

    result = {}
    for idx, row in businesses.iterrows():
        result[row["business_id"]] = matrix[idx]

    logger.info("Built tabular features: %d restaurants × %d features", len(result), matrix.shape[1])
    return result, matrix.shape[1]


def combine_embeddings(
    business_ids: List[str],
    clip_embs: Dict[str, np.ndarray],
    text_embs: Dict[str, np.ndarray],
    tabular_embs: Dict[str, np.ndarray],
    clip_dim: int = 512,
    text_dim: int = 384,
    tabular_dim: int = 64,
) -> tuple[np.ndarray, List[str]]:
    tab_actual = 0
    for v in tabular_embs.values():
        tab_actual = len(v)
        break

    valid_ids = []
    rows = []
    for biz_id in business_ids:
        clip_vec = clip_embs.get(biz_id, np.zeros(clip_dim, dtype=np.float32))
        text_vec = text_embs.get(biz_id, np.zeros(text_dim, dtype=np.float32))
        tab_vec = tabular_embs.get(biz_id, np.zeros(tab_actual, dtype=np.float32))
        row = np.concatenate([clip_vec, text_vec, tab_vec]).astype(np.float32)
        rows.append(row)
        valid_ids.append(biz_id)

    matrix = np.stack(rows)
    logger.info(
        "Combined embedding matrix: %d restaurants × %d dims (clip=%d + text=%d + tab=%d)",
        matrix.shape[0], matrix.shape[1], clip_dim, text_dim, tab_actual,
    )
    return matrix, valid_ids


def save_embeddings(
    embeddings: np.ndarray,
    business_ids: List[str],
    output_dir: str | Path,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "restaurant_features.npy", embeddings)
    pd.Series(business_ids).to_csv(out / "restaurant_ids.csv", index=False, header=False)
    logger.info("Saved embeddings to %s", out)


def load_embeddings(output_dir: str | Path) -> tuple[np.ndarray, List[str]]:
    out = Path(output_dir)
    embeddings = np.load(out / "restaurant_features.npy")
    ids = pd.read_csv(out / "restaurant_ids.csv", header=None)[0].tolist()
    return embeddings, ids

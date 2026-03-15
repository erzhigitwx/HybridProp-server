from __future__ import annotations

import logging
import pandas as pd
from ml.config import Config
logger = logging.getLogger(__name__)


def temporal_split(
    user_history: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_ratio = cfg.data.val_ratio
    test_ratio = cfg.data.test_ratio

    dates_sorted = user_history["date"].sort_values()
    n = len(dates_sorted)
    train_end = dates_sorted.iloc[int(n * (1 - val_ratio - test_ratio))]
    val_end = dates_sorted.iloc[int(n * (1 - test_ratio))]

    train = user_history[user_history["date"] <= train_end].copy()
    val = user_history[(user_history["date"] > train_end) & (user_history["date"] <= val_end)].copy()
    test = user_history[user_history["date"] > val_end].copy()

    train_users = set(train["user_id"].unique())
    train_items = set(train["business_id"].unique())
    val = val[val["user_id"].isin(train_users) & val["business_id"].isin(train_items)]
    test = test[test["user_id"].isin(train_users) & test["business_id"].isin(train_items)]

    logger.info(
        "Temporal split: train=%d, val=%d, test=%d | cutoffs: %s / %s",
        len(train), len(val), len(test),
        train_end.date(), val_end.date(),
    )
    return train, val, test

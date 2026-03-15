import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)

from ml.config import load_config
from ml.data.splits import temporal_split
from ml.data.dataset import RecInteractionDataset, EvalDataset, build_dataloaders
from ml.embeddings.aggregator import load_embeddings
from ml.models.two_tower import TwoTowerModel
from ml.training.trainer import Trainer, HistoryCollator
from ml.evaluation.evaluator import Evaluator
from ml.utils.device import get_device, seed_everything


def _build_user_histories(
    train_df: pd.DataFrame,
    item2idx: Dict[str, int],
) -> Dict[int, List[Tuple[int, float]]]:
    from ml.data.dataset import RecInteractionDataset

    histories: Dict[str, List[Tuple[str, float, pd.Timestamp]]] = defaultdict(list)
    for _, row in train_df.iterrows():
        histories[row["user_id"]].append(
            (row["business_id"], row["stars"], row["date"])
        )

    return histories


def main():
    parser = argparse.ArgumentParser(description="Train two-tower model")
    parser.add_argument("--config", type=str, default="ml/config/config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    logging.info("Loading preprocessed data...")
    businesses = pd.read_parquet(f"{cfg.paths.processed}/businesses.parquet")
    user_history = pd.read_parquet(f"{cfg.paths.processed}/user_history.parquet")
    features, business_ids = load_embeddings(cfg.paths.embeddings)

    train_df, val_df, test_df = temporal_split(user_history, cfg)

    biz_to_city = dict(zip(businesses["business_id"], businesses["city"]))

    train_ds = RecInteractionDataset(
        interactions=train_df,
        all_business_ids=business_ids,
        business_to_city=biz_to_city,
        positive_threshold=cfg.data.positive_threshold,
        hard_negative_ratio=cfg.model.loss.hard_negative_ratio,
    )

    user_hist_raw: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for _, row in train_df.sort_values("date").iterrows():
        uidx = train_ds.user2idx.get(row["user_id"])
        iidx = train_ds.item2idx.get(row["business_id"])
        if uidx is not None and iidx is not None:
            user_hist_raw[uidx].append((iidx, row["stars"]))

    collator = HistoryCollator(
        user_histories=user_hist_raw,
        max_history=cfg.model.user_tower.max_history,
    )

    train_loader = __import__("torch.utils.data", fromlist=["DataLoader"]).DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )
    eval_users = []
    train_interactions: Dict[int, Set[int]] = defaultdict(set)
    for uidx, items in train_ds.user_all.items():
        train_interactions[uidx] = items

    for _, group in val_df[val_df["stars"] >= cfg.data.positive_threshold].groupby("user_id"):
        uid = group.iloc[0]["user_id"]
        uidx = train_ds.user2idx.get(uid)
        if uidx is None:
            continue
        gt_items = {train_ds.item2idx[bid] for bid in group["business_id"] if bid in train_ds.item2idx}
        if gt_items:
            eval_users.append((uidx, gt_items))

    evaluator = Evaluator(
        cfg=cfg,
        eval_users=eval_users,
        user_histories=user_hist_raw,
        train_interactions=train_interactions,
        max_history=cfg.model.user_tower.max_history,
    )

    model = TwoTowerModel(cfg.model, features)
    logging.info(
        "Model parameters: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=None,
        evaluator=evaluator,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()

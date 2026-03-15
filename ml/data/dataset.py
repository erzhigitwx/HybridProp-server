from __future__ import annotations
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ml.config import Config

logger = logging.getLogger(__name__)


class RecInteractionDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        all_business_ids: List[str],
        business_to_city: Dict[str, str],
        positive_threshold: float = 4.0,
        hard_negative_ratio: float = 0.3,
        num_negatives: int = 1,
    ):
        self.positive_threshold = positive_threshold
        self.hard_negative_ratio = hard_negative_ratio
        self.num_negatives = num_negatives

        self.user_ids = sorted(interactions["user_id"].unique())
        self.item_ids = sorted(all_business_ids)
        self.user2idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item2idx = {iid: i for i, iid in enumerate(self.item_ids)}
        self.idx2item = {i: iid for iid, i in self.item2idx.items()}

        self.user_positives: Dict[int, List[int]] = defaultdict(list)
        self.user_all: Dict[int, Set[int]] = defaultdict(set)
        self.city_items: Dict[str, List[int]] = defaultdict(list)

        for _, row in interactions.iterrows():
            uidx = self.user2idx.get(row["user_id"])
            iidx = self.item2idx.get(row["business_id"])
            if uidx is None or iidx is None:
                continue
            self.user_all[uidx].add(iidx)
            if row["stars"] >= positive_threshold:
                self.user_positives[uidx].append(iidx)

        for iid, city in business_to_city.items():
            if iid in self.item2idx:
                self.city_items[city].append(self.item2idx[iid])

        self.item_to_city = {}
        for iid, city in business_to_city.items():
            if iid in self.item2idx:
                self.item_to_city[self.item2idx[iid]] = city

        self.pairs: List[Tuple[int, int]] = []
        for uidx, pos_items in self.user_positives.items():
            for pidx in pos_items:
                self.pairs.append((uidx, pidx))

        logger.info(
            "Dataset: %d users, %d items, %d positive pairs",
            len(self.user_ids), len(self.item_ids), len(self.pairs),
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[int, int, int]:
        uidx, pidx = self.pairs[index]
        nidx = self._sample_negative(uidx, pidx)
        return uidx, pidx, nidx

    def _sample_negative(self, uidx: int, pidx: int) -> int:
        interacted = self.user_all[uidx]
        all_items = len(self.item_ids)

        if np.random.random() < self.hard_negative_ratio:
            city = self.item_to_city.get(pidx)
            if city and city in self.city_items:
                candidates = [i for i in self.city_items[city] if i not in interacted]
                if candidates:
                    return candidates[np.random.randint(len(candidates))]

        for _ in range(100):
            neg = np.random.randint(all_items)
            if neg not in interacted:
                return neg
        return np.random.randint(all_items)

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_items(self) -> int:
        return len(self.item_ids)


class EvalDataset(Dataset):
    def __init__(
        self,
        eval_interactions: pd.DataFrame,
        user2idx: Dict[str, int],
        item2idx: Dict[str, int],
        positive_threshold: float = 4.0,
    ):
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.entries: List[Tuple[int, List[int]]] = []

        grouped = eval_interactions[eval_interactions["stars"] >= positive_threshold].groupby("user_id")
        for uid, group in grouped:
            uidx = user2idx.get(uid)
            if uidx is None:
                continue
            gt = [item2idx[bid] for bid in group["business_id"] if bid in item2idx]
            if gt:
                self.entries.append((uidx, gt))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[int, List[int]]:
        return self.entries[index]


def build_dataloaders(
    train_ds: RecInteractionDataset,
    val_ds: Optional[EvalDataset],
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_eval_collate,
        )
    return train_loader, val_loader


def _eval_collate(batch):
    user_idxs = [b[0] for b in batch]
    ground_truths = [b[1] for b in batch]
    return user_idxs, ground_truths

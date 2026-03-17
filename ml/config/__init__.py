from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class PathsConfig:
    raw_data: str = "data/yelp_dataset"
    photos_dir: str = "data/yelp_photos/photos"
    photos_json: str = "data/yelp_photos/photos.json"
    processed: str = "data/processed"
    embeddings: str = "data/embeddings"
    checkpoints: str = "checkpoints"
    logs: str = "logs"


@dataclass
class DataConfig:
    min_reviews: int = 5
    min_photos: int = 1
    max_reviews_per_business: int = 30
    min_user_interactions: int = 3
    positive_threshold: float = 4.0
    restaurant_categories: List[str] = field(default_factory=lambda: ["Restaurants", "Food"])
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class CLIPConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    dim: int = 512
    batch_size: int = 64
    image_size: int = 224
    photo_weights: Dict[str, float] = field(
        default_factory=lambda: {"food": 1.5, "inside": 1.0, "outside": 0.8, "drink": 1.2, "menu": 0.3}
    )


@dataclass
class TextEmbConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    batch_size: int = 128
    max_seq_length: int = 512


@dataclass
class TabularEmbConfig:
    input_dim: int = 96
    hidden_dim: int = 64
    output_dim: int = 64


@dataclass
class EmbeddingsConfig:
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    text: TextEmbConfig = field(default_factory=TextEmbConfig)
    tabular: TabularEmbConfig = field(default_factory=TabularEmbConfig)


@dataclass
class TowerConfig:
    input_dim: int = 960
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    output_dim: int = 256
    dropout: float = 0.2
    activation: str = "gelu"
    batch_norm: bool = True


@dataclass
class UserTowerConfig:
    output_dim: int = 256
    attention_heads: int = 4
    max_history: int = 50
    dropout: float = 0.1


@dataclass
class LossConfig:
    type: str = "bpr"
    infonce_temperature: float = 0.07
    num_negatives: int = 10
    hard_negative_ratio: float = 0.3


@dataclass
class ModelConfig:
    restaurant_tower: TowerConfig = field(default_factory=TowerConfig)
    user_tower: UserTowerConfig = field(default_factory=UserTowerConfig)
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class SchedulerConfig:
    type: str = "cosine_warmup"
    warmup_ratio: float = 0.05
    eta_min: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    patience: int = 7
    min_delta: float = 0.001
    metric: str = "ndcg@10"


@dataclass
class CheckpointConfig:
    save_every: int = 1
    keep_top_k: int = 3


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


@dataclass
class EvaluationConfig:
    ks: List[int] = field(default_factory=lambda: [5, 10, 20])
    metrics: List[str] = field(default_factory=lambda: ["hit_rate", "ndcg", "mrr"])
    num_eval_users: int = 5000


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "restaurants"
    vector_dim: int = 256
    distance: str = "Cosine"
    batch_size: int = 500
    api_key: Optional[str] = None


@dataclass
class SoftFilterConfig:
    default_softness: float = 0.15
    min_strict_ratio: float = 0.6
    filter_bonus: float = 0.1


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    default_top_k: int = 20
    soft_filter: SoftFilterConfig = field(default_factory=SoftFilterConfig)


@dataclass
class LoggingConfig:
    backend: str = "wandb"
    wandb_project: str = "hybridprop-recsys"
    log_every_n_steps: int = 50
    log_recommendations_every: int = 5


@dataclass
class Config:
    project_name: str = "hybridprop-recsys"
    seed: int = 42
    device: str = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _resolve_type(dc_class, f):
    import dataclasses
    import typing
    try:
        hints = typing.get_type_hints(dc_class)
        return hints.get(f.name, f.type)
    except Exception:
        return f.type


def _merge(dc_class, raw: dict):
    import dataclasses as dc
    if raw is None:
        return dc_class()
    kwargs = {}
    for f in dc.fields(dc_class):
        if f.name not in raw:
            continue
        val = raw[f.name]
        resolved = _resolve_type(dc_class, f)
        if dc.is_dataclass(resolved) and isinstance(val, dict):
            val = _merge(resolved, val)
        kwargs[f.name] = val
    return dc_class(**kwargs)


def load_config(path: Optional[str] = None) -> Config:
    if path is None:
        path = os.environ.get(
            "HYBRIDPROP_CONFIG",
            str(Path(__file__).parent / "config.yaml"),
        )
    p = Path(path)
    if not p.exists():
        return Config()

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    top = {}
    if "project" in raw:
        top["project_name"] = raw["project"].get("name", "hybridprop-recsys")
        top["seed"] = raw["project"].get("seed", 42)
        top["device"] = raw["project"].get("device", "auto")
    for key in ("paths", "data", "embeddings", "model", "training",
                "evaluation", "qdrant", "api", "logging"):
        if key in raw:
            top[key] = raw[key]

    return _merge(Config, top)
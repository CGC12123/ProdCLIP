import os
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str = "data/cleaned_data.csv"
    image_dir: str = "data/images/"  # Images are in the data/images directory
    splits: dict = None  # Will be set to {"train": 0.7, "val": 0.15, "test": 0.15}

    def __post_init__(self):
        if self.splits is None:
            self.splits = {"train": 0.7, "val": 0.15, "test": 0.15}


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "openai/clip-vit-large-patch14"
    # model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 77
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Training configuration"""
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = None

    # Training parameters
    epochs: int = 60
    learning_rate: float = 1e-4
    temperature: float = 0.07
    batch_size: int = 512
    save_path: str = "./outputs/lora_adapter_l14_r16"
    save_every_n_epochs: int = 10  # Save checkpoint every n epochs

    # Hard negative training parameters
    lambda_hard: float = 1
    hard_negative_type: str = "category"  # "batch" or "category"
    # lora_path: str = None # None for no use
    lora_path: str = 'outputs/lora_adapter_l14_r16/checkpoint_epoch_60' # None for no use

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

@dataclass
class LoRAConfig:
    """LoRA specific configuration"""
    use_lora: bool = True
    lora_path: str = "outputs/lora_adapter_l14_r16/checkpoint_epoch_60"


@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    embedding_dim: int = 768  # CLIP embedding dimension for ViT-L/14
    # embedding_dim: int = 512  # CLIP embedding dimension for ViT-B/32
    cache_dir: str = "cache/clip-vit-large-patch14/"
    # cache_dir: str = "cache/clip-vit-base-patch32/"
    embeddings_file: str = "image_embeddings.npy"
    id_to_path_file: str = "id_to_path.json"


@dataclass
class IndexConfig:
    """Index configuration"""
    index_file: str = "cache/clip-vit-large-patch14/faiss_index.bin"  # Put it in the cache directory
    metric_type: str = "l2"  # We'll use Inner Product (cosine similarity) but embeddings are normalized


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    recall_at_k: list = None  # Will be set to [1, 5]

    def __post_init__(self):
        if self.recall_at_k is None:
            self.recall_at_k = [1, 5]


@dataclass
class AnalysisConfig:
    """Analysis configuration"""
    plots_output_dir: str = "outputs/lora_adapter_l14_r16/plots/"
    plot_embedding_dist: bool = True
    plot_similarity_dist: bool = True


@dataclass
class DistillationConfig:
    """Knowledge Distillation configuration"""
    # Model configurations
    teacher_model_name: str = "openai/clip-vit-large-patch14"
    student_model_name: str = "openai/clip-vit-base-patch32"

    # Teacher LoRA configuration (for using fine-tuned teacher)
    teacher_use_lora: bool = True
    teacher_lora_path: str = "outputs/lora_adapter_l14_r16/checkpoint_epoch_60"

    # Distillation types: list of methods to use ("embedding", "similarity", "logits")
    distill_types: list = None  # Will be set to ["embedding", "similarity"]

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 1e-5
    temperature: float = 1.0
    lambda_embed: float = 0.5
    lambda_similarity: float = 0.5
    lambda_logits: float = 0.5  # logits distillation
    epochs: int = 50
    save_path: str = "./outputs/distilled_model"

    # Checkpoint and saving
    checkpoint_interval: int = 5  # Save checkpoints every N epochs
    resume_from_checkpoint: str = None

    # Additional parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.distill_types is None:
            self.distill_types = ["embedding", "similarity"]  # Default distillation types


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    lora: LoRAConfig = None
    embedding: EmbeddingConfig = None
    index: IndexConfig = None
    evaluation: EvaluationConfig = None
    analysis: AnalysisConfig = None
    distillation: DistillationConfig = None

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.index is None:
            self.index = IndexConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.analysis is None:
            self.analysis = AnalysisConfig()
        if self.distillation is None:
            self.distillation = DistillationConfig()


# Global config instance
config = Config()
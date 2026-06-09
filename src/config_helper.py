import os
import yaml
from pydantic import BaseModel, Field

class PipelineConfig(BaseModel):
    target_score: int
    max_iterations: int
    min_score_improvement: int = 3
    early_stop_patience: int = 3
    cumulative_training: bool = True

class ModelConfig(BaseModel):
    base_model: str
    embedding_model: str
    device: str
    use_4bit: bool
    max_seq_length: int = 8192
    generation_max_tokens: int = 4096

class PathConfig(BaseModel):
    data_raw: str
    data_processed: str
    vectordb: str
    sft_dataset: str
    outputs_generated: str
    outputs_reports: str
    checkpoints: str

class RagConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    top_k: int

class TrainingConfig(BaseModel):
    learning_rate: float
    num_epochs: int
    batch_size: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.3

class AppSettings(BaseModel):
    pipeline: PipelineConfig
    model: ModelConfig
    paths: PathConfig
    rag: RagConfig
    training: TrainingConfig

def load_settings(config_path: str = "config/settings.yaml") -> AppSettings:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppSettings(**data)

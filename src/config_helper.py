import os
import yaml
from pydantic import BaseModel, Field

class PipelineConfig(BaseModel):
    target_score: int
    max_iterations: int

class ModelConfig(BaseModel):
    base_model: str
    embedding_model: str
    device: str
    use_4bit: bool

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

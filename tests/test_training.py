import os
import pytest
import shutil
from src.config_helper import load_settings
from src.training.dataset_builder import DatasetBuilder
from src.training.trainer import QLoraTrainer
from src.training.orchestrator import TrainingOrchestrator

@pytest.fixture(scope="module")
def setup_training():
    settings = load_settings()
    settings.paths.sft_dataset = "data/test_sft_dataset"
    settings.paths.checkpoints = "outputs/test_checkpoints"
    settings.model.device = "cpu" # mock mode force
    
    db = DatasetBuilder(settings)
    trainer = QLoraTrainer(settings)
    orch = TrainingOrchestrator(settings, trainer)
    
    yield db, trainer, orch, settings
    
    # Cleanup
    if os.path.exists("data/test_sft_dataset"):
        shutil.rmtree("data/test_sft_dataset")
    if os.path.exists("outputs/test_checkpoints"):
        shutil.rmtree("outputs/test_checkpoints")

def test_dataset_building_and_orchestration(setup_training):
    db, trainer, orch, settings = setup_training
    
    # 1. Dataset building
    report = {
        "total_score": 70,
        "metrics": {"numerical_accuracy": 60, "format_compliance": 80, "linguistic_appropriateness": 75},
        "justification": "Incorrect dosage.",
        "error_tokens": ["5mg"]
    }
    
    db.append_and_update(
        ctd_md="## Dosage\n5mg was specified.",
        true_if="## Interview Form\nCorrect dosage is 5.0mg.",
        true_siori="## Siori\nCorrect dosage is 5.0mg.",
        report=report
    )
    
    assert os.path.exists(db.dataset_file)
    
    # 2. Hyperparameter dynamic scheduling checks
    assert orch.consecutive_failures == 0
    lr1, ep1 = orch.determine_training_parameters()
    
    orch.consecutive_failures = 1
    lr2, ep2 = orch.determine_training_parameters()
    assert lr2 < lr1
    assert ep2 > ep1
    
    # 3. Training integration trigger
    new_adapter = orch.handle_feedback(report, db.dataset_file, iteration=1)
    assert new_adapter is not None
    assert "gemma-2-27b-qlora-iter-1" in new_adapter

import os
from typing import Any

class TrainingOrchestrator:
    def __init__(self, settings: Any, trainer: Any):
        self.settings = settings
        self.trainer = trainer
        self.consecutive_failures = 0
        self.active_adapter = None

    def determine_training_parameters(self) -> tuple[float, int]:
        """Dynamically adjusts hyperparameters based on consecutive failures to target score."""
        base_lr = self.settings.training.learning_rate
        base_epochs = self.settings.training.num_epochs

        if self.consecutive_failures == 0:
            return base_lr, base_epochs
        elif self.consecutive_failures == 1:
            # Drop learning rate slightly and increase epoch to consolidate knowledge
            return base_lr * 0.5, base_epochs + 2
        else:
            # Deeper fine-tuning for harder mistakes
            return base_lr * 0.25, base_epochs + 4

    def handle_feedback(self, score_report: dict, dataset_path: str, iteration: int) -> str | None:
        """Evaluates score results and triggers dynamic QLoRA learning when below target score."""
        target = self.settings.pipeline.target_score
        score = score_report.get("total_score", 0)
        
        if score >= target:
            print(f"[Orchestrator] Score {score} met target {target}. Resetting failure count.")
            self.consecutive_failures = 0
            return None
            
        self.consecutive_failures += 1
        print(f"[Orchestrator] Score {score} is below target {target}. Triggering autonomous fine-tuning iteration.")
        
        lr, epochs = self.determine_training_parameters()
        new_adapter = self.trainer.train(dataset_path, iteration, lr, epochs)
        self.active_adapter = new_adapter
        return new_adapter

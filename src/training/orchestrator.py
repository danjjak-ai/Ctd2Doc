import os
import json
import time
from typing import Any, Optional


class TrainingOrchestrator:
    def __init__(self, settings: Any, trainer: Any):
        self.settings = settings
        self.trainer = trainer
        self.consecutive_failures = 0
        self.active_adapter = None
        self.best_score = 0.0
        self.best_adapter_path: Optional[str] = None
        self.score_history: list[float] = []
        self.plateau_count = 0
        self.training_log: list[dict] = []

    def determine_training_parameters(self, current_score: float = 0) -> tuple[float, int]:
        """점수대별 적응적 하이퍼파라미터 결정.

        낮은 점수에서는 공격적으로 학습하고, 높은 점수에서는 미세 조정.
        """
        base_lr = self.settings.training.learning_rate
        base_epochs = self.settings.training.num_epochs

        if current_score < 40:
            # 초기 단계: 공격적 학습 (높은 LR, 많은 에포크)
            lr, epochs = base_lr, base_epochs + 3
            strategy = "aggressive"
        elif current_score < 60:
            # 중간 단계: 표준 학습
            lr, epochs = base_lr * 0.75, base_epochs + 1
            strategy = "standard"
        elif current_score < 80:
            # 후반 단계: 섬세한 학습 (낮은 LR)
            lr, epochs = base_lr * 0.5, base_epochs
            strategy = "refined"
        else:
            # 최종 단계: 미세 조정 (매우 낮은 LR, 긴 에포크)
            lr, epochs = base_lr * 0.1, base_epochs + 2
            strategy = "fine-tune"

        # 연속 실패 시 추가 보정
        if self.consecutive_failures >= 3:
            lr *= 0.5
            epochs += 2
            strategy += "+deep"

        print(f"[Orchestrator] Strategy: {strategy} | LR: {lr:.6f} | Epochs: {epochs}")
        return lr, epochs

    def detect_plateau(self) -> bool:
        """점수 정체를 감지.

        최근 N회 이터레이션에서 점수 향상이 min_score_improvement 미만이면 정체로 판단.
        """
        min_improvement = self.settings.pipeline.min_score_improvement
        patience = self.settings.pipeline.early_stop_patience

        if len(self.score_history) < 2:
            return False

        recent_improvement = self.score_history[-1] - self.score_history[-2]

        if recent_improvement < min_improvement:
            self.plateau_count += 1
            print(f"[Orchestrator] Plateau detected ({self.plateau_count}/{patience}). "
                  f"Improvement: {recent_improvement:.1f} < {min_improvement}")
        else:
            self.plateau_count = 0

        return self.plateau_count >= patience

    def handle_feedback(self, score_report: dict, dataset_path: str,
                        iteration: int) -> Optional[str]:
        """평가 결과를 분석하고 필요 시 QLoRA 학습을 트리거.

        Returns:
            새 어댑터 경로 (학습 수행 시) 또는 None (목표 달성 시)
        """
        target = self.settings.pipeline.target_score
        score = score_report.get("total_score", 0)

        # 점수 이력 기록
        self.score_history.append(score)

        # Best score 트래킹
        if score > self.best_score:
            self.best_score = score
            print(f"[Orchestrator] New best score: {score}")

        if score >= target:
            print(f"[Orchestrator] Score {score} met target {target}. Resetting failure count.")
            self.consecutive_failures = 0
            self.plateau_count = 0
            return None

        # 정체 감지
        if self.detect_plateau():
            print(f"[Orchestrator] Early stopping triggered. Reverting to best adapter: {self.best_adapter_path}")
            return self.best_adapter_path

        self.consecutive_failures += 1
        print(f"[Orchestrator] Score {score} below target {target}. "
              f"Consecutive failures: {self.consecutive_failures}. "
              f"Triggering autonomous fine-tuning.")

        # 적응적 하이퍼파라미터 결정
        lr, epochs = self.determine_training_parameters(current_score=score)

        # 학습 실행
        new_adapter = self.trainer.train(dataset_path, iteration, lr, epochs)

        # Best adapter 업데이트
        self.active_adapter = new_adapter
        if score >= self.best_score:
            self.best_adapter_path = new_adapter

        # 학습 로그 기록
        self._log_training(iteration, score, lr, epochs, new_adapter)

        return new_adapter

    def _log_training(self, iteration: int, score: float, lr: float,
                      epochs: int, adapter_path: str):
        """학습 이력을 JSON 로그로 기록."""
        entry = {
            "timestamp": time.time(),
            "iteration": iteration,
            "score_before_training": score,
            "learning_rate": lr,
            "epochs": epochs,
            "adapter_path": adapter_path,
            "consecutive_failures": self.consecutive_failures,
            "plateau_count": self.plateau_count,
            "best_score": self.best_score,
        }
        self.training_log.append(entry)

        # 파일에도 기록
        log_dir = self.settings.paths.outputs_reports
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training_log.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_summary(self) -> dict:
        """학습 오케스트레이션 요약 정보를 반환."""
        return {
            "total_iterations": len(self.score_history),
            "best_score": self.best_score,
            "best_adapter_path": self.best_adapter_path,
            "current_score": self.score_history[-1] if self.score_history else 0,
            "consecutive_failures": self.consecutive_failures,
            "plateau_count": self.plateau_count,
            "score_history": self.score_history,
        }

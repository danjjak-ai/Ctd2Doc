"""간단 검증 스크립트: 모든 모듈의 import와 기본 기능 테스트."""
import sys
import os

def test_config():
    print("=== 1. Config Test ===")
    from src.config_helper import load_settings
    s = load_settings()
    assert s.model.base_model == "google/gemma-4-E4B-it"
    assert s.model.max_seq_length == 8192
    assert s.pipeline.early_stop_patience == 3
    assert s.training.gradient_accumulation_steps == 4
    print(f"  Model: {s.model.base_model}")
    print(f"  PASS")
    return s

def test_model_registry():
    print("\n=== 2. Model Registry Test ===")
    from src.model_registry import (
        get_model_preset, build_chat_messages,
        build_sft_chat_text, get_lora_target_modules,
        supports_system_prompt,
    )
    p = get_model_preset("google/gemma-4-E4B-it")
    assert p["family"] == "gemma4"
    assert p["supports_system_prompt"] is True
    assert "q_proj" in p["lora_target_modules"]

    msgs = build_chat_messages("google/gemma-4-E4B-it", "System prompt", "User input")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"

    sft = build_sft_chat_text("google/gemma-4-E4B-it", "Sys", "Input", "Output")
    assert len(sft) == 3
    assert sft[2]["role"] == "model"

    assert supports_system_prompt("google/gemma-4-E4B-it") is True
    assert supports_system_prompt("google/gemma-2-27b-it") is False

    modules = get_lora_target_modules("google/gemma-4-E4B-it")
    assert len(modules) == 7
    print(f"  Family: {p['family']}, System prompt: {p['supports_system_prompt']}")
    print(f"  PASS")

def test_gpu_utils():
    print("\n=== 3. GPU Utils Test ===")
    from src.gpu_utils import get_gpu_memory_info, print_gpu_status
    info = get_gpu_memory_info()
    print(f"  GPU available: {info['available']}")
    print_gpu_status("Test")
    print(f"  PASS")

def test_dataset_builder(settings):
    print("\n=== 4. Dataset Builder Test ===")
    from src.training.dataset_builder import DatasetBuilder
    builder = DatasetBuilder(settings)
    sample = builder.build_sft_sample(
        "CTD sample text",
        "True IF document",
        {"justification": "test", "error_tokens": ["5mg"]},
    )
    assert "instruction" in sample
    assert "input" in sample
    assert "output" in sample
    assert "5mg" in sample["instruction"]
    stats = builder.get_dataset_stats()
    print(f"  Dataset stats: {stats}")
    print(f"  PASS")

def test_orchestrator(settings):
    print("\n=== 5. Orchestrator Test ===")
    from src.training.orchestrator import TrainingOrchestrator
    from src.training.trainer import QLoraTrainer
    trainer = QLoraTrainer(settings)
    orch = TrainingOrchestrator(settings, trainer)

    # 적응적 하이퍼파라미터
    lr1, ep1 = orch.determine_training_parameters(30)
    lr2, ep2 = orch.determine_training_parameters(70)
    lr3, ep3 = orch.determine_training_parameters(82)
    assert lr1 > lr2 > lr3, f"LR should decrease: {lr1}, {lr2}, {lr3}"
    print(f"  Score 30 -> LR={lr1:.6f}, Epochs={ep1}")
    print(f"  Score 70 -> LR={lr2:.6f}, Epochs={ep2}")
    print(f"  Score 82 -> LR={lr3:.6f}, Epochs={ep3}")

    # 정체 감지
    orch.score_history = [50, 51, 51.5, 51.8]
    assert orch.detect_plateau() is False  # 첫 호출, count=1
    orch.score_history.append(52)
    orch.detect_plateau()  # count=2
    orch.score_history.append(52.1)
    result = orch.detect_plateau()  # count=3 → patience 도달
    assert result is True
    print(f"  Plateau detection: PASS")

    summary = orch.get_summary()
    assert "best_score" in summary
    print(f"  PASS")

def test_scoring(settings):
    print("\n=== 6. Scoring Test ===")
    from src.evaluation.scoring import ScoringSystem
    scoring = ScoringSystem(settings)
    trend = scoring.get_score_trend()
    assert "scores" in trend
    print(f"  Score trend keys: {list(trend.keys())}")
    print(f"  PASS")

def test_judge(settings):
    print("\n=== 7. Judge Agent Test ===")
    from src.evaluation.judge_agent import JudgeAgent
    judge = JudgeAgent(settings)

    # Rule-based 폴백 평가 테스트
    gt = "副作用 12.0% 用法 5.0mg 用量"
    gen = "副作用 12.0% 用法 5.0mg 用量"
    report = judge.evaluate(gen, gt)
    assert "total_score" in report
    assert "error_categories" in report
    print(f"  Score (perfect match): {report['total_score']}")

    # 듀얼 평가 테스트
    dual = judge.evaluate_dual(gen, gt, gen, gt)
    assert "if_score" in dual
    assert "siori_score" in dual
    assert "sub_reports" in dual
    print(f"  Dual score: {dual['total_score']} (IF: {dual['if_score']}, Siori: {dual['siori_score']})")
    print(f"  PASS")

def test_main_cli():
    print("\n=== 8. Main CLI Test ===")
    # argparse 테스트
    sys.argv = ["main.py", "--dry-run", "--config", "config/settings.yaml"]
    from main import parse_args
    args = parse_args()
    assert args.dry_run is True
    assert args.config == "config/settings.yaml"
    assert args.single_drug is None
    print(f"  CLI args: dry_run={args.dry_run}, config={args.config}")
    print(f"  PASS")
    # 원래 argv 복원
    sys.argv = ["main.py"]

def main():
    print("CTD2Doc Module Verification\n")

    settings = test_config()
    test_model_registry()
    test_gpu_utils()
    test_dataset_builder(settings)
    test_orchestrator(settings)
    test_scoring(settings)
    test_judge(settings)
    test_main_cli()

    print("\n" + "=" * 50)
    print("  ALL TESTS PASSED!")
    print("=" * 50)

if __name__ == "__main__":
    main()

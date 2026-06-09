"""CTD2Doc: 자율 개선 파이프라인 CLI 진입점.

Usage:
    python main.py                          # 전체 약품 리스트 실행
    python main.py --config path/to/yaml    # 커스텀 설정 파일 사용
    python main.py --dry-run                # Mock 모드 드라이런
    python main.py --single-drug 10023      # 단일 약품만 실행
    python main.py --status                 # 점수 이력 요약 출력
"""
import argparse
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="CTD2Doc: Autonomous Self-Improving Pipeline for Medical Document Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/settings.yaml",
        help="설정 파일 경로 (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="CPU Mock 모드로 파이프라인 플로우만 검증 (GPU 불필요)",
    )
    parser.add_argument(
        "--single-drug", type=str, default=None,
        help="특정 약품 코드만 실행 (예: --single-drug 10023)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="최대 반복 횟수 오버라이드",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="점수 이력 요약을 출력하고 종료",
    )
    return parser.parse_args()


def run_status(config_path: str):
    """기존 점수 이력 요약만 출력."""
    from src.config_helper import load_settings
    from src.evaluation.scoring import ScoringSystem

    settings = load_settings(config_path)
    scoring = ScoringSystem(settings)
    scoring.print_summary()

    history = scoring.get_history()
    if not history:
        print("아직 평가 이력이 없습니다.")
    else:
        # 약품별 요약
        drug_codes = list(set(h["japic_code"] for h in history))
        for code in drug_codes:
            scoring.print_summary(code)


def run_pipeline(args):
    """메인 파이프라인 실행."""
    from src.config_helper import load_settings

    # 설정 로드
    settings = load_settings(args.config)

    # Dry-run 모드: device를 CPU로 강제 전환
    if args.dry_run:
        print("\n🔧 DRY-RUN MODE: Forcing CPU (mock) mode for pipeline validation.\n")
        settings.model.device = "cpu"

    # Max iterations 오버라이드
    if args.max_iterations:
        settings.pipeline.max_iterations = args.max_iterations

    # 파이프라인 초기화
    from src.pipeline import AutonomousPipeline
    pipeline = AutonomousPipeline(settings=settings)

    # 단일 약품 모드
    if args.single_drug:
        # static_drugs에서 해당 코드 찾기
        drug = next(
            (d for d in pipeline.crawler.static_drugs
             if d["japic_code"] == args.single_drug),
            None,
        )
        if drug is None:
            # 코드로 찾지 못하면 임시 drug 딕셔너리 생성
            drug = {"japic_code": args.single_drug, "name_ja": f"Drug_{args.single_drug}"}
            print(f"[Main] Drug code '{args.single_drug}' not in static list. "
                  f"Using generated entry.")

        result = pipeline.run_single_drug(drug, args.max_iterations)
        print(f"\n[Main] Final best score: {result['score']}")
    else:
        # 전체 실행
        pipeline.run_all()


def main():
    args = parse_args()

    if args.status:
        run_status(args.config)
        return

    run_pipeline(args)


if __name__ == "__main__":
    main()

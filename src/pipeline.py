import os
import time
from typing import Any, Dict, List, Optional

from src.config_helper import load_settings
from src.ingestion.crawler import CrawlerAgent
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.preprocessor import Preprocessor
from src.rag.embedder import Embedder
from src.rag.vectorstore import VectorStore
from src.rag.retriever import Retriever
from src.inference.generator import Generator
from src.evaluation.judge_agent import JudgeAgent
from src.evaluation.scoring import ScoringSystem
from src.training.dataset_builder import DatasetBuilder
from src.training.trainer import QLoraTrainer
from src.training.orchestrator import TrainingOrchestrator
from src.gpu_utils import free_gpu_memory, print_gpu_status


class AutonomousPipeline:
    """자율 개선 파이프라인: 추론 → 듀얼 평가 → 조건부 학습 반복.

    Gemma 4 E4B-IT 모델을 사용하여 CTD 데이터로부터 의약품 문서(IF, Siori)를
    생성하고, Judge Agent 평가 결과에 따라 QLoRA 추가 학습을 자동 수행합니다.
    """

    def __init__(self, config_path: str = "config/settings.yaml", settings: Any = None):
        if settings is not None:
            self.settings = settings
        else:
            self.settings = load_settings(config_path)

        print(f"\n{'='*60}")
        print(f"  CTD2Doc Autonomous Pipeline")
        print(f"  Model: {self.settings.model.base_model}")
        print(f"  Target Score: {self.settings.pipeline.target_score}")
        print(f"  Max Iterations: {self.settings.pipeline.max_iterations}")
        print(f"{'='*60}\n")

        # 1. Ingestion Components
        self.crawler = CrawlerAgent(self.settings)
        self.parser = PDFParser(self.settings)
        self.preprocessor = Preprocessor(self.settings)

        # 2. RAG Components
        self.embedder = Embedder(self.settings)
        self.vectorstore = VectorStore(self.settings)
        self.retriever = Retriever(self.vectorstore, self.embedder, self.settings)

        # 3. Gen & Eval Components
        self.generator = Generator(self.settings)
        self.judge = JudgeAgent(self.settings)
        self.scoring = ScoringSystem(self.settings)

        # 4. Training Components
        self.dataset_builder = DatasetBuilder(self.settings)
        self.trainer = QLoraTrainer(self.settings)
        self.orchestrator = TrainingOrchestrator(self.settings, self.trainer)

    def run_single_drug_pipeline(self, drug: Dict[str, Any],
                                 iteration: int) -> Dict[str, Any]:
        """단일 약품에 대한 생성 → 듀얼 평가 → 조건부 학습 루프 1회 실행.

        Returns:
            score, report, generated_if, generated_siori, early_stop 포함 딕셔너리
        """
        code = drug["japic_code"]
        name = drug.get("name_ja", "Unknown")
        target_score = self.settings.pipeline.target_score

        print(f"\n{'─'*50}")
        print(f"  [Pipeline] Iteration {iteration} | Drug: {name} ({code})")
        print(f"{'─'*50}")
        start_time = time.time()

        # Step 1: Download raw documents
        raw_results = self.crawler.collect([drug])
        drug_data = next((d for d in raw_results if d["japic_code"] == code), None)
        if not drug_data:
            raise ValueError(f"Failed to crawl/locate drug data for code: {code}")

        # Step 2: PDF Parsing to Markdown (CTD)
        ctd_md = ""
        if drug_data.get("ctd_pdf"):
            ctd_md = self.parser.parse(drug_data["ctd_pdf"])
        else:
            ctd_md = self.parser._run_basic_fallback("mock_ctd.pdf")

        # Step 3: Extract ground truths
        gt_data = self.preprocessor.extract_targets(drug_data)
        true_if = gt_data["true_if"]
        true_siori = gt_data["true_siori"]

        # Step 4: Index CTD chunks into RAG
        if ctd_md:
            self.vectorstore.index_document(ctd_md, code, self.embedder)

        # Step 5: Inference (Generate IF and Siori)
        print_gpu_status("Before inference")
        generated_if = self.generator.generate(self.retriever, "interview_form", drug)
        generated_siori = self.generator.generate(self.retriever, "siori", drug)

        # Step 6: 듀얼 평가 (IF + Siori 가중 합산)
        report = self.judge.evaluate_dual(
            generated_if, true_if,
            generated_siori, true_siori,
            if_weight=0.6, siori_weight=0.4,
        )

        current_score = report.get("total_score", 0)

        # Log evaluation metrics
        self.scoring.log_score(code, iteration, report)

        # Step 7: 조건부 학습 (목표 미달 시)
        early_stop = False
        if current_score < target_score:
            print(f"\n  [Pipeline] Score {current_score} < Target {target_score}. "
                  f"Entering training phase...")

            # 학습 데이터셋에 실패 데이터 추가
            self.dataset_builder.append_and_update(ctd_md, true_if, true_siori, report)

            # GPU 메모리 관리: 추론 모델 언로드 → 학습 → 재로드
            self.generator.unload_model()
            free_gpu_memory()
            print_gpu_status("After inference model unload")

            # Orchestrator를 통한 적응적 학습
            new_adapter = self.orchestrator.handle_feedback(
                score_report=report,
                dataset_path=self.dataset_builder.dataset_file,
                iteration=iteration,
            )

            # 학습 후 추론 모델 재로드
            free_gpu_memory()
            self.generator.reload_model()

            # 새 어댑터 적용
            if new_adapter:
                self.generator.reload_adapter(new_adapter)

            # 조기 종료 체크 (정체 감지)
            if self.orchestrator.detect_plateau():
                print(f"\n  [Pipeline] ⚠ Plateau detected! "
                      f"Reverting to best adapter and moving to next drug.")
                if self.orchestrator.best_adapter_path:
                    self.generator.reload_adapter(self.orchestrator.best_adapter_path)
                early_stop = True
        else:
            print(f"\n  [Pipeline] ✅ Goal achieved for {name}! "
                  f"Score: {current_score} >= {target_score}")

        # Step 8: 생성 결과 저장
        self._save_outputs(code, iteration, generated_if, generated_siori)

        elapsed = time.time() - start_time
        print(f"\n  [Pipeline] Iteration {iteration} completed in {elapsed:.1f}s. "
              f"Score: {current_score}")

        return {
            "score": current_score,
            "report": report,
            "generated_if": generated_if,
            "generated_siori": generated_siori,
            "early_stop": early_stop,
        }

    def run_all(self):
        """설정된 약품 리스트 전체에 대해 반복 학습 루프를 순차 실행."""
        pipeline_start = time.time()
        results_summary: List[Dict[str, Any]] = []

        for drug_idx, drug in enumerate(self.crawler.static_drugs):
            code = drug["japic_code"]
            name = drug.get("name_ja", "Unknown")

            print(f"\n{'='*60}")
            print(f"  Drug [{drug_idx + 1}/{len(self.crawler.static_drugs)}]: {name} ({code})")
            print(f"{'='*60}")

            # Orchestrator 상태 리셋 (약품별 독립 학습)
            self.orchestrator.consecutive_failures = 0
            self.orchestrator.plateau_count = 0

            iteration = 1
            max_iters = self.settings.pipeline.max_iterations
            best_score_for_drug = 0

            while iteration <= max_iters:
                res = self.run_single_drug_pipeline(drug, iteration)
                best_score_for_drug = max(best_score_for_drug, res["score"])

                # 목표 달성 시 다음 약품으로 이동
                if res["score"] >= self.settings.pipeline.target_score:
                    break

                # 조기 종료 (정체 감지)
                if res.get("early_stop", False):
                    print(f"  [Pipeline] Early stopping for {name} after {iteration} iterations.")
                    break

                iteration += 1

            # 약품별 점수 요약
            self.scoring.print_summary(code)
            results_summary.append({
                "drug": name,
                "japic_code": code,
                "best_score": best_score_for_drug,
                "iterations": iteration,
                "target_met": best_score_for_drug >= self.settings.pipeline.target_score,
            })

        # 전체 파이프라인 요약
        total_elapsed = time.time() - pipeline_start
        self._print_final_summary(results_summary, total_elapsed)

    def run_single_drug(self, drug: Dict[str, Any],
                        max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """단일 약품에 대해서만 반복 학습 루프를 실행 (CLI --single-drug 용)."""
        max_iters = max_iterations or self.settings.pipeline.max_iterations
        code = drug["japic_code"]
        name = drug.get("name_ja", "Unknown")

        print(f"\n[Pipeline] Running single drug mode: {name} ({code})")

        iteration = 1
        best_result = None

        while iteration <= max_iters:
            res = self.run_single_drug_pipeline(drug, iteration)

            if best_result is None or res["score"] > best_result["score"]:
                best_result = res

            if res["score"] >= self.settings.pipeline.target_score:
                break
            if res.get("early_stop", False):
                break

            iteration += 1

        self.scoring.print_summary(code)
        return best_result

    def _save_outputs(self, japic_code: str, iteration: int,
                      generated_if: str, generated_siori: str):
        """생성된 문서를 파일로 저장."""
        gen_dir = os.path.join(
            self.settings.paths.outputs_generated,
            f"JapicCode_{japic_code}",
        )
        os.makedirs(gen_dir, exist_ok=True)

        # 이터레이션별 저장 (히스토리 보존)
        with open(os.path.join(gen_dir, f"generated_if_iter_{iteration}.md"),
                  "w", encoding="utf-8") as f:
            f.write(generated_if)
        with open(os.path.join(gen_dir, f"generated_siori_iter_{iteration}.md"),
                  "w", encoding="utf-8") as f:
            f.write(generated_siori)

        # 최신 버전도 저장 (덮어쓰기)
        with open(os.path.join(gen_dir, "generated_if.md"),
                  "w", encoding="utf-8") as f:
            f.write(generated_if)
        with open(os.path.join(gen_dir, "generated_siori.md"),
                  "w", encoding="utf-8") as f:
            f.write(generated_siori)

    def _print_final_summary(self, results: List[Dict[str, Any]], elapsed: float):
        """전체 파이프라인 실행 결과 요약 출력."""
        print(f"\n{'='*60}")
        print(f"  Pipeline Execution Complete")
        print(f"  Total Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"{'='*60}")
        print(f"\n  {'Drug':<20} {'Best Score':<12} {'Iters':<8} {'Status'}")
        print(f"  {'─'*55}")
        for r in results:
            status = "✅ Target Met" if r["target_met"] else "⚠ Below Target"
            print(f"  {r['drug']:<20} {r['best_score']:<12} {r['iterations']:<8} {status}")

        met_count = sum(1 for r in results if r["target_met"])
        print(f"\n  Summary: {met_count}/{len(results)} drugs met target score")
        print(f"  Orchestrator: {self.orchestrator.get_summary()}")
        print()

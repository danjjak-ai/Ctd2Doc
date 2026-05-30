import os
import pytest
import shutil
from src.config_helper import load_settings
from src.inference.generator import Generator
from src.evaluation.judge_agent import JudgeAgent
from src.evaluation.scoring import ScoringSystem

from src.rag.embedder import Embedder
from src.rag.vectorstore import VectorStore
from src.rag.retriever import Retriever

@pytest.fixture(scope="module")
def setup_inference_eval():
    settings = load_settings()
    settings.model.device = "cpu"
    settings.paths.outputs_reports = "outputs/test_reports"
    settings.paths.vectordb = "data/test_vectordb_infer"
    
    # Initialize basic pipeline objects for testing context dependency
    embedder = Embedder(settings)
    vectorstore = VectorStore(settings)
    retriever = Retriever(vectorstore, embedder, settings)
    
    # Populate mock collections so retriever fetches successfully
    vectorstore.index_document(
        markdown_text="## Dosage\nAdult dose is 5.0mg. Nausea was seen in 5.0% of participants.",
        japic_code="10023",
        embedder=embedder
    )
    
    generator = Generator(settings)
    judge = JudgeAgent(settings)
    scoring = ScoringSystem(settings)
    
    yield generator, judge, scoring, retriever
    
    # Cleanup
    if os.path.exists("outputs/test_reports"):
        try:
            shutil.rmtree("outputs/test_reports")
        except PermissionError:
            pass
    if os.path.exists("data/test_vectordb_infer"):
        try:
            shutil.rmtree("data/test_vectordb_infer")
        except PermissionError:
            pass


def test_inference_and_judging(setup_inference_eval):
    generator, judge, scoring, retriever = setup_inference_eval
    drug = {"japic_code": "10023", "name_ja": "TestDrug"}
    
    # 1. Generate Document
    generated_if = generator.generate(retriever, "interview_form", drug)
    assert len(generated_if) > 0
    assert "1024" in generated_if or "5.0mg" in generated_if # Mock target checks
    
    # 2. Judge Generation
    ground_truth = """臨床試験において、総症例1024例中、副作用が報告された。
- 悪心・嘔吐 (Nausea): 51例 (5.0%)
- 頭痛: 12.0%
- 用量: 5.0mg
"""
    report = judge.evaluate(generated_if, ground_truth)
    assert "total_score" in report
    assert "metrics" in report
    assert report["metrics"]["numerical_accuracy"] > 0
    
    # 3. Log Score
    scoring.log_score("10023", iteration=1, report=report)
    history = scoring.get_history()
    assert len(history) == 1
    assert history[0]["japic_code"] == "10023"

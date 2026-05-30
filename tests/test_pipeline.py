import os
import pytest
import shutil
from src.config_helper import load_settings
from src.pipeline import AutonomousPipeline

@pytest.fixture(scope="module")
def setup_pipeline():
    settings = load_settings()
    settings.model.device = "cpu"
    settings.paths.data_raw = "data/test_raw_pipeline"
    settings.paths.data_processed = "data/test_processed_pipeline"
    settings.paths.vectordb = "data/test_vectordb_pipeline"
    settings.paths.sft_dataset = "data/test_sft_pipeline"
    settings.paths.outputs_generated = "outputs/test_generated_pipeline"
    settings.paths.outputs_reports = "outputs/test_reports_pipeline"
    settings.paths.checkpoints = "outputs/test_checkpoints_pipeline"
    
    # Force single iteration limit for test speed
    settings.pipeline.max_iterations = 1
    
    pipeline = AutonomousPipeline(settings=settings)
    
    yield pipeline
    
    # Cleanup temp dirs
    for path in [settings.paths.data_raw, settings.paths.data_processed, 
                 settings.paths.vectordb, settings.paths.sft_dataset, 
                 settings.paths.outputs_generated, settings.paths.outputs_reports, 
                 settings.paths.checkpoints]:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except PermissionError:
                pass


def test_pipeline_e2e_flow(setup_pipeline):
    pipeline = setup_pipeline
    
    # Mocking crawler download inside test settings
    drug = {
        "japic_code": "10023",
        "name_ja": "TestDrugPipeline",
        "pmda_url": "https://www.pmda.go.jp/iyaku/10023",
        "radar_url": "https://www.rad-ar.or.jp/detail/10023"
    }
    
    # Run single drug loop sequence
    res = pipeline.run_single_drug_pipeline(drug, iteration=1)
    
    assert "score" in res
    assert "report" in res
    assert "generated_if" in res
    assert "generated_siori" in res
    
    # Verify outputs are written to respective folders
    assert os.path.exists(os.path.join(pipeline.settings.paths.outputs_generated, "JapicCode_10023", "generated_if.md"))
    assert os.path.exists(os.path.join(pipeline.settings.paths.outputs_reports, "score_history.jsonl"))

# Unit tests for Ingestion Agent components (Crawler, PDFParser, Preprocessor)
import os
import pytest
from src.config_helper import load_settings
from src.ingestion.crawler import CrawlerAgent
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.preprocessor import Preprocessor

@pytest.fixture
def test_settings():
    return load_settings()

def test_crawler_init(test_settings):
    crawler = CrawlerAgent(test_settings)
    assert crawler.mode in ["static", "auto", "both"]
    assert len(crawler.static_drugs) > 0

def test_pdf_parser_fallback(test_settings):
    parser = PDFParser(test_settings)
    # Testing parser behavior with non-existent file or fallbacks
    markdown_result = parser.parse("non_existent_file.pdf")
    assert markdown_result == ""
    
    # Testing standard structure simulation
    simulated_result = parser._run_basic_fallback("dummy.pdf")
    assert "COMMON TECHNICAL DOCUMENT" in simulated_result
    assert "표 2-1" in simulated_result or "Table" in simulated_result or "부작용" in simulated_result

def test_preprocessor(test_settings):
    preprocessor = Preprocessor(test_settings)
    drug_data = {
        "japic_code": "10023",
        "ctd_pdf": None,
        "if_pdf": None,
        "siori_html": None
    }
    extracted = preprocessor.extract_targets(drug_data)
    assert "true_if" in extracted
    assert "true_siori" in extracted

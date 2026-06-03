import os
import sys
from src.config_helper import load_settings
from src.ingestion.crawler import CrawlerAgent
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.preprocessor import Preprocessor

def main():
    print("=== [Test] Ingestion & Preprocessing Test for イムブルビカ ===")
    
    # 1. Load Settings
    settings = load_settings("config/settings.yaml")
    
    # Define drug details for イムブルビカ
    drug = {
        "japic_code": "4291043",
        "name_ja": "イムブルビカ",
        "pmda_url": "https://www.pmda.go.jp/PmdaSearch/iyakuDetail/GeneralList/4291043",
        "radar_url": "https://www.rad-ar.or.jp/siori/search/result?n=42369"
    }
    
    # 2. Ingestion - Crawler Agent
    print("\n[Step 1] Crawler Agent Initialization")
    # Initialize CrawlerAgent with mock settings where static_drugs has イムブルビカ
    # We will override crawler's static_drugs to only include イムブルビカ for this test
    crawler = CrawlerAgent(settings)
    crawler.static_drugs = [drug]
    crawler.mode = "static" # only crawl static_drugs
    
    print("Running crawler.collect()...")
    collected = crawler.collect()
    print(f"Collected results: {collected}")
    
    # 3. PDF Parsing
    print("\n[Step 2] PDF Parsing")
    parser = PDFParser(settings)
    drug_data = collected[0]
    
    parsed_ctd = ""
    if drug_data.get("ctd_pdf"):
        parsed_ctd = parser.parse(drug_data["ctd_pdf"])
        print(f"CTD PDF parsed successfully! Characters count: {len(parsed_ctd)}")
    else:
        print("No CTD PDF was downloaded, using fallback.")
        parsed_ctd = parser._run_basic_fallback("mock_ctd.pdf")
        
    parsed_if = ""
    if drug_data.get("if_pdf"):
        parsed_if = parser.parse(drug_data["if_pdf"])
        print(f"IF PDF parsed successfully! Characters count: {len(parsed_if)}")
    else:
        print("No IF PDF was downloaded.")
        
    # 4. Preprocessing
    print("\n[Step 3] Preprocessing (Extracting Targets)")
    preprocessor = Preprocessor(settings)
    extracted = preprocessor.extract_targets(drug_data)
    
    print("\n[Test Summary]")
    print(f"Japic Code: {drug_data['japic_code']}")
    print(f"Drug Name: {drug_data['name']}")
    print(f"CTD PDF Path: {drug_data['ctd_pdf']}")
    print(f"IF PDF Path: {drug_data['if_pdf']}")
    print(f"Siori HTML Path: {drug_data['siori_html']}")
    print(f"Extracted Ground Truth Siori Characters: {len(extracted.get('true_siori', ''))}")
    print(f"Extracted Ground Truth IF Characters: {len(extracted.get('true_if', ''))}")

if __name__ == "__main__":
    main()

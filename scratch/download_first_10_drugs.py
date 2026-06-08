import os
import shutil
from src.config_helper import load_settings
from src.ingestion.crawler import CrawlerAgent
from src.ingestion.master_downloader import MasterDownloader

def main():
    print("=== Downloading Master File and Fetching Documents for First 10 Drugs ===")
    
    settings = load_settings("config/settings.yaml")
    
    # 1. Force regeneration of the mock master file to get all 10 entries
    downloader = MasterDownloader("config/drug_list.yaml")
    y_txt_path = os.path.join(downloader.download_dir, "y.txt")
    if os.path.exists(y_txt_path):
        os.remove(y_txt_path)
    
    # Trigger download (which generates the 10 mock entries if live download is HTML index)
    y_txt_path = downloader.download()
    print(f"Master file downloaded/generated at: {y_txt_path}")
    
    # 2. Parse the first 10 drugs from y.txt
    records = downloader.parse_master(y_txt_path)
    first_10_records = records[:10]
    
    print(f"Found {len(first_10_records)} drugs in the master file:")
    for idx, r in enumerate(first_10_records):
        print(f"  {idx+1}. {r['name_ja']} (YJ: {r['yj_code']}, JAPIC: {r['japic_code']})")
        
    # 3. Create target drug list for CrawlerAgent
    target_drugs = []
    for r in first_10_records:
        # Construct standard URLs
        pmda_url = f"https://www.pmda.go.jp/PmdaSearch/iyakuDetail/GeneralList/{r['japic_code']}"
        radar_url = f"https://www.rad-ar.or.jp/kusuri/search/detail/{r['japic_code']}"
        
        target_drugs.append({
            "yj_code": r["yj_code"],
            "japic_code": r["japic_code"],
            "name_ja": r["name_ja"],
            "pmda_url": pmda_url,
            "radar_url": radar_url
        })
        
    # 4. Initialize CrawlerAgent and run document collection
    crawler = CrawlerAgent(settings, "config/drug_list.yaml")
    crawler.static_drugs = target_drugs
    crawler.mode = "static"  # Only crawl our specified 10 drugs
    
    print("\nStarting Crawler for the 10 drugs...")
    results = crawler.collect()
    
    print("\n=== Document Download Summary ===")
    for r in results:
        print(f"\nDrug: {r['name']} (JAPIC: {r['japic_code']})")
        print(f"  - CTD PDF: {r['ctd_pdf']}")
        print(f"  - IF PDF: {r['if_pdf']}")
        print(f"  - Siori HTML: {r['siori_html']}")
        print(f"  - Total CTD PDFs: {len(r['all_ctd_pdfs'])}")
        print(f"  - Total Package Inserts: {len(r['all_package_inserts'])}")
        print(f"  - Total Patient Guides: {len(r['all_patient_guides'])}")

if __name__ == "__main__":
    main()

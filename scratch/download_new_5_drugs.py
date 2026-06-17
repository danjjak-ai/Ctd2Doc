import os
from src.config_helper import load_settings
from src.ingestion.crawler import CrawlerAgent
from src.ingestion.master_downloader import MasterDownloader

def main():
    print("=== Downloading/Generating Master File and Fetching Documents for the 5 New Drugs ===")
    
    settings = load_settings("config/settings.yaml")
    
    # 1. Initialize downloader
    downloader = MasterDownloader("config/drug_list.yaml")
    
    # Trigger download/mock generation
    y_txt_path = downloader.download()
    print(f"Master file verified at: {y_txt_path}")
    
    # 2. Parse the records
    records = downloader.parse_master(y_txt_path)
    print(f"Total records in master: {len(records)}")
    
    # Retrieve the last 5 records
    new_5_records = records[-5:]
    
    print(f"Found the 5 new drugs:")
    for idx, r in enumerate(new_5_records):
        print(f"  {idx+1}. {r['name_ja']} (YJ: {r['yj_code']}, JAPIC: {r['japic_code']})")
        
    # 3. Create target drug list for CrawlerAgent
    target_drugs = []
    for r in new_5_records:
        target_drugs.append({
            "yj_code": r["yj_code"],
            "japic_code": r["japic_code"],
            "name_ja": r["name_ja"],
            "pmda_url": f"https://www.pmda.go.jp/PmdaSearch/rdSearch/02/{r['yj_code']}?user=1",
            "radar_url": f"https://www.rad-ar.or.jp/kusuri/search/detail/{r['japic_code']}"
        })
        
    # 4. Initialize CrawlerAgent and run document collection
    crawler = CrawlerAgent(settings, "config/drug_list.yaml")
    
    print("\nStarting Crawler for the 5 new drugs...")
    results = crawler.collect(target_drugs)
    
    print("\n=== Document Download Summary ===")
    for r in results:
        print(f"\nDrug: {r['name']} (JAPIC: {r['japic_code']})")
        print(f"  - CTD PDF: {r['ctd_pdf']}")
        print(f"  - IF PDF: {r['if_pdf']}")
        print(f"  - Siori HTML: {r['siori_html']}")

if __name__ == "__main__":
    main()

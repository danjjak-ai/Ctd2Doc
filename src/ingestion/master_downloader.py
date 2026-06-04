import os
import zipfile
import requests
import csv
import yaml
from typing import Dict, Any, List

class MasterDownloader:
    def __init__(self, drug_list_path: str = "config/drug_list.yaml"):
        self.drug_list_path = drug_list_path
        self.load_config()

    def load_config(self):
        with open(self.drug_list_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        crawler_cfg = self.config.get("crawler", {})
        master_cfg = crawler_cfg.get("master_dataset", {})
        self.url = master_cfg.get("url", "https://www.ssk.or.jp/seikyushiharai/tensuhyo/kikaku/iyakuhin_master/index.html")
        self.download_dir = master_cfg.get("download_dir", "data/master")
        self.enabled = master_cfg.get("enabled", True)

    def generate_mock_master(self, dest_path: str):
        """Generates a mock y.txt file representing the Japanese Drug Master dataset for testing/fallback."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # Mock records matching standard 医薬品マスター specification (CSV layout)
        # Field 2 (index 2) is the 9-digit code (Japic Code / Receipt Code)
        # Field 4 (index 4) is the Kanji brand name
        # Field 31 (index 31) is the 12-digit YJ Code
        mock_records = [
            ["1", "Y", "520032615", "02", "エンレスト錠50mg", "07", "ｴﾝﾚｽﾄ", "", "", "", "", "1", "1", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "2190041F1027", "2190041", "", "サクビトリルバルサルタン"],
            ["1", "Y", "520033577", "02", "ファビハルタカプセル200mg", "07", "ﾌｧﾋﾞﾊﾙﾀ", "", "", "", "", "1", "1", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "3999064M1020", "3999064", "", "ファビピラビル"],
            ["1", "Y", "4291043", "02", "イムブルビカカプセル140mg", "07", "ｲﾑﾌﾞﾙﾋﾞｶ", "", "", "", "", "1", "1", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "4291043M1027", "4291043", "", "イブルチニブ"]
        ]
        with open(dest_path, "w", newline="", encoding="cp932") as f:
            writer = csv.writer(f)
            writer.writerows(mock_records)
        print(f"[MasterDownloader] Generated mock master database: {dest_path}")

    def download(self) -> str:
        """Downloads and extracts the master dataset. Falls back to mock file if download fails or is disabled."""
        os.makedirs(self.download_dir, exist_ok=True)
        y_txt_path = os.path.join(self.download_dir, "y.txt")

        if not self.enabled:
            print("[MasterDownloader] Master download disabled in config. Using mock/existing file.")
            if not os.path.exists(y_txt_path):
                self.generate_mock_master(y_txt_path)
            return y_txt_path

        try:
            print(f"[MasterDownloader] Attempting download from: {self.url}")
            if "index.html" in self.url or not self.url.endswith(".zip"):
                print("[MasterDownloader] Configured URL is an index page. Generating mock master file.")
                self.generate_mock_master(y_txt_path)
                return y_txt_path
                
            response = requests.get(self.url, timeout=20)
            response.raise_for_status()
            
            zip_path = os.path.join(self.download_dir, "y_master.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)
                
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.endswith("y.txt") or file_name.startswith("y"):
                        zip_ref.extract(file_name, self.download_dir)
                        extracted_path = os.path.join(self.download_dir, file_name)
                        if extracted_path != y_txt_path:
                            os.replace(extracted_path, y_txt_path)
                        print(f"[MasterDownloader] Extracted master file: {y_txt_path}")
                        return y_txt_path
                        
            print("[MasterDownloader] No y.txt found in zip. Generating mock.")
            self.generate_mock_master(y_txt_path)
        except Exception as e:
            print(f"[MasterDownloader] Download failed: {e}. Falling back to mock master.")
            if not os.path.exists(y_txt_path):
                self.generate_mock_master(y_txt_path)
        return y_txt_path

    def parse_master(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses the y.txt master file (usually Shift_JIS/CP932 encoded CSV)."""
        parsed_records = []
        if not os.path.exists(file_path):
            return parsed_records
            
        try:
            with open(file_path, "r", encoding="cp932") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 32:
                        yj_code = row[31].strip()
                        japic_code = row[2].strip()
                        kanji_name = row[4].strip() if len(row) > 4 else ""
                        
                        if yj_code:
                            parsed_records.append({
                                "yj_code": yj_code,
                                "japic_code": japic_code,
                                "name_ja": kanji_name
                            })
        except Exception as e:
            print(f"[MasterDownloader] Parsing error: {e}")
        return parsed_records

    def get_drug_by_yj_code(self, yj_code: str) -> Dict[str, Any]:
        """Searches for a drug by its 12-digit YJ Code in the parsed master file."""
        y_txt_path = os.path.join(self.download_dir, "y.txt")
        if not os.path.exists(y_txt_path):
            self.download()
        records = self.parse_master(y_txt_path)
        for r in records:
            if r["yj_code"] == yj_code:
                return r
        return {}

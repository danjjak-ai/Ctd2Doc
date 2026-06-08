import re
import os
import time
import requests
import chardet
import urllib.parse
from bs4 import BeautifulSoup
from typing import Dict, List, Any
import yaml
from src.ingestion.master_downloader import MasterDownloader

class CrawlerAgent:
    def __init__(self, settings: Any, drug_list_path: str = "config/drug_list.yaml"):
        self.settings = settings
        self.drug_list_path = drug_list_path
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Initialize MasterDownloader
        self.master_downloader = MasterDownloader(self.drug_list_path)
        
        # Maintain public attributes for test compatibility
        self.mode = "both"
        self.static_drugs = []
        self._load_static_drugs_from_master()

    def _load_static_drugs_from_master(self):
        """Populates self.static_drugs from master dataset for test compatibility."""
        try:
            y_txt_path = self.master_downloader.download()
            records = self.master_downloader.parse_master(y_txt_path)
            for r in records:
                self.static_drugs.append({
                    "yj_code": r["yj_code"],
                    "japic_code": r["japic_code"],
                    "name_ja": r["name_ja"],
                    "pmda_url": f"https://www.pmda.go.jp/PmdaSearch/rdSearch/02/{r['yj_code']}?user=1",
                    "radar_url": f"https://www.rad-ar.or.jp/kusuri/search/detail/{r['japic_code']}"
                })
        except Exception:
            pass

    def fetch_page(self, url: str) -> str:
        """Fetches a page, automatically handling Japanese character encoding."""
        response = self.session.get(url, timeout=15)
        response.raise_for_status()
        
        detected = chardet.detect(response.content)
        encoding = detected.get("encoding", "utf-8")
        if encoding and encoding.lower() in ["shift_jis", "shift-jis", "cp932", "euc-jp"]:
            try:
                return response.content.decode(encoding)
            except Exception:
                return response.content.decode("utf-8", errors="ignore")
        return response.text

    def get_pmda_details_url(self, yj_code: str) -> str:
        """Finds the actual PMDA detail page by following the official YJ Code redirect URL."""
        redirect_url = f"https://www.pmda.go.jp/PmdaSearch/rdSearch/02/{yj_code}?user=1"
        try:
            res = self.session.get(redirect_url, timeout=15)
            res.raise_for_status()
            return res.url
        except Exception as e:
            print(f"[Warning] Failed to resolve redirect for YJ Code {yj_code}: {e}")
            return f"https://www.pmda.go.jp/PmdaSearch/iyakuDetail/GeneralList/{yj_code[:7]}"

    def parse_pmda_details(self, search_url: str) -> Dict[str, List[str]]:
        """Scrapes the resolved PMDA page to extract all 6 categories of document links."""
        urls = {
            "ctd": [],
            "package_insert": [],
            "patient_guide": [],
            "if": [],
            "siori": [],
            "rmp": [],
            "rmp_material": [],
        }
        if not search_url:
            return urls
            
        try:
            html = self.fetch_page(search_url)
            soup = BeautifulSoup(html, "html.parser")
            
            all_a = soup.find_all("a", href=True)
            for a_tag in all_a:
                href = a_tag["href"]
                text = a_tag.get_text().strip().replace('\n', '')
                full_url = urllib.parse.urljoin(search_url, href)
                
                if "ResultDataSetPDF" in href:
                    urls["package_insert"].append(full_url)
                elif "GUI" in href or "患者向医薬品ガイド" in text or "患者向ガイド" in text or text.startswith("G_"):
                    urls["patient_guide"].append(full_url)
                elif "interview" in href.lower() or "interview" in text.lower() or text.startswith("IF_") or "インタビューフォーム" in text:
                    urls["if"].append(full_url)
                elif "RMP" in href or "RMP" in text:
                    if "RMPm" in href or "資材" in text or "適正使用" in text or "ガイド" in text:
                        urls["rmp_material"].append(full_url)
                    else:
                        urls["rmp"].append(full_url)
                elif "ctd" in href.lower() or "common technical document" in text.lower() or "審査報告" in text or "報告書" in text or "drugs/" in href:
                    urls["ctd"].append(full_url)

            # Extract RAD-AR siori link from GeneralList page if not directly present
            general_list_url = None
            for a_tag in all_a:
                href = a_tag["href"]
                if "iyakuDetail/GeneralList/" in href:
                    general_list_url = urllib.parse.urljoin(search_url, href)
                    break
            
            if general_list_url:
                try:
                    gen_html = self.fetch_page(general_list_url)
                    gen_soup = BeautifulSoup(gen_html, "html.parser")
                    for a_tag in gen_soup.find_all("a", href=True):
                        href = a_tag["href"]
                        text = a_tag.get_text()
                        if "rad-ar.or.jp" in href or "くすりのしおり" in text:
                            urls["siori"].append(urllib.parse.urljoin(general_list_url, href))
                except Exception as ex:
                    print(f"  [Warning] Failed to fetch general list page for siori link: {ex}")

        except Exception as e:
            print(f"[Warning] Failed to parse PMDA detail page {search_url}: {e}")
        return urls

    def parse_radar_siori(self, radar_url: str) -> str:
        """Scrapes Kusuri-no-Siori (くすりのしおり) page from RAD-AR site."""
        try:
            html = self.fetch_page(radar_url)
            soup = BeautifulSoup(html, "html.parser")
            content_div = soup.find("div", class_="siori-content") or soup.find("div", id="siori_area")
            if content_div:
                return str(content_div)
            return html
        except Exception as e:
            print(f"[Warning] Failed to parse RAD-AR Siori url {radar_url}: {e}")
            return ""

    def download_file(self, url: str, dest_path: str):
        """Downloads file with rate-limiting."""
        if not url:
            return
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        time.sleep(1.0)
        
        response = self.session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def collect(self, targets: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Main ingress collection loop.
        Downloads the master dataset (y.txt).
        If targets is provided, crawls those specific drugs.
        Otherwise, reads the first 10 drugs from y.txt and crawls them.
        """
        # Ensure master dataset is downloaded/mocked
        y_txt_path = self.master_downloader.download()
        
        if targets is None:
            # Load the drugs directly from the master file (y.txt)
            records = self.master_downloader.parse_master(y_txt_path)
            target_records = records[:10]
            print(f"[Crawler] Loaded {len(target_records)} target drugs directly from master file {y_txt_path}")
            
            targets = []
            for r in target_records:
                targets.append({
                    "yj_code": r["yj_code"],
                    "japic_code": r["japic_code"],
                    "name_ja": r["name_ja"],
                    "pmda_url": f"https://www.pmda.go.jp/PmdaSearch/rdSearch/02/{r['yj_code']}?user=1",
                    "radar_url": f"https://www.rad-ar.or.jp/kusuri/search/detail/{r['japic_code']}"
                })
        else:
            print(f"[Crawler] Processing {len(targets)} specified target drugs")

        collected_results = []
        for drug in targets:
            yj_code = drug.get("yj_code")
            japic_code = drug.get("japic_code")
            name = drug.get("name_ja", "Unknown")
            
            # Auto-resolve codes
            if not yj_code and japic_code:
                try:
                    records = self.master_downloader.parse_master(y_txt_path)
                    for r in records:
                        if r["japic_code"] == japic_code:
                            yj_code = r["yj_code"]
                            break
                except Exception:
                    pass
            if yj_code and not japic_code:
                japic_code = yj_code[:7]

            if not japic_code:
                continue

            raw_dir = os.path.join(self.settings.paths.data_raw, f"JapicCode_{japic_code}")
            os.makedirs(raw_dir, exist_ok=True)
            
            print(f"[Crawler] Processing: {name} (YJ: {yj_code}, JAPIC: {japic_code})")
            
            # Determine target search page
            detail_url = self.get_pmda_details_url(yj_code) if yj_code else drug.get("pmda_url")
            
            # Scrape links
            doc_urls = self.parse_pmda_details(detail_url)
            
            # Fallback to config URLs if parsed links are empty
            if not doc_urls["siori"] and drug.get("radar_url"):
                doc_urls["siori"].append(drug["radar_url"])
            if not doc_urls["package_insert"] and drug.get("pmda_url") and not yj_code:
                doc_urls["package_insert"].append(drug["pmda_url"])
            
            downloaded = {
                "ctd_pdf": [],
                "package_insert": [],
                "patient_guide": [],
                "if_pdf": [],
                "siori_html": [],
                "rmp": [],
                "rmp_material": []
            }
            
            # 1. Download CTD
            for idx, url in enumerate(doc_urls["ctd"][:3]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"ctd{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["ctd_pdf"].append(dest)
                except Exception as e:
                    print(f"  [Error] Downloading CTD failed: {e}")
            
            # 2. Download 添付文書 (Package Insert)
            for idx, url in enumerate(doc_urls["package_insert"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"package_insert{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["package_insert"].append(dest)
                except Exception as e:
                    print(f"  [Error] Downloading Package Insert failed: {e}")
            
            # 3. Download 患者向医薬品ガイド (Patient Guide)
            for idx, url in enumerate(doc_urls["patient_guide"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"patient_guide{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["patient_guide"].append(dest)
                except Exception as e:
                    print(f"  [Error] Downloading Patient Guide failed: {e}")
                    
            # 4. Download IF (Interview Form)
            for idx, url in enumerate(doc_urls["if"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"interview_form{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["if_pdf"].append(dest)
                except Exception as e:
                    print(f"  [Error] Downloading IF failed: {e}")
                    
            # 5. Download くすりのしおり (Siori)
            for idx, url in enumerate(doc_urls["siori"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                siori_html = self.parse_radar_siori(url)
                if siori_html:
                    dest = os.path.join(raw_dir, f"siori{suffix}.html")
                    with open(dest, "w", encoding="utf-8") as f:
                        f.write(siori_html)
                    downloaded["siori_html"].append(dest)
            
            # 6. Download RMP
            for idx, url in enumerate(doc_urls["rmp"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"rmp{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["rmp"].append(dest)
                except Exception as e:
                    print(f"  [Error] Downloading RMP failed: {e}")
                    
            # 7. Download RMP資材 (RMP Materials)
            for idx, url in enumerate(doc_urls["rmp_material"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"rmp_material{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["rmp_material"].append(dest)
                except Exception as e:
                    print(f"  [Error] Downloading RMP Material failed: {e}")
            
            print(f"  Downloaded: CTDs={len(downloaded['ctd_pdf'])}, Inserts={len(downloaded['package_insert'])}, Guides={len(downloaded['patient_guide'])}, IFs={len(downloaded['if_pdf'])}, Sioris={len(downloaded['siori_html'])}")
            
            collected_results.append({
                "yj_code": yj_code,
                "japic_code": japic_code,
                "name": name,
                "ctd_pdf": downloaded["ctd_pdf"][0] if downloaded["ctd_pdf"] else None,
                "if_pdf": downloaded["if_pdf"][0] if downloaded["if_pdf"] else None,
                "siori_html": downloaded["siori_html"][0] if downloaded["siori_html"] else None,
                "all_ctd_pdfs": downloaded["ctd_pdf"],
                "all_package_inserts": downloaded["package_insert"],
                "all_patient_guides": downloaded["patient_guide"],
                "all_if_pdfs": downloaded["if_pdf"],
                "all_siori_htmls": downloaded["siori_html"],
                "all_rmps": downloaded["rmp"],
                "all_rmp_materials": downloaded["rmp_material"]
            })
            
        return collected_results

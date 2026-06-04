import re
import os
import time
import requests
import chardet
from bs4 import BeautifulSoup
from typing import Dict, List, Any
import yaml

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
        # Establish session cookies by visiting search page first
        try:
            self.session.get("https://www.pmda.go.jp/PmdaSearch/iyakuSearch/", timeout=10)
        except Exception as e:
            print(f"[Warning] Failed to pre-visit PMDA search page: {e}")
        self.load_drug_list()

    def load_drug_list(self):
        with open(self.drug_list_path, "r", encoding="utf-8") as f:
            self.drug_config = yaml.safe_load(f)
        self.mode = self.drug_config.get("crawler", {}).get("mode", "both")
        self.static_drugs = self.drug_config.get("static_list", [])
        self.auto_categories = self.drug_config.get("crawler", {}).get("auto_categories", [])

    def fetch_page(self, url: str) -> str:
        """Fetches a page, automatically handling Japanese character encoding like Shift-JIS or EUC-JP."""
        response = self.session.get(url, timeout=15)
        response.raise_for_status()
        
        # Detect encoding if not UTF-8
        detected = chardet.detect(response.content)
        encoding = detected.get("encoding", "utf-8")
        if encoding and encoding.lower() in ["shift_jis", "shift-jis", "cp932", "euc-jp"]:
            try:
                return response.content.decode(encoding)
            except Exception:
                return response.content.decode("utf-8", errors="ignore")
        return response.text

    def parse_pmda_details(self, japic_code: str, search_url: str) -> Dict[str, List[str]]:
        """Scrapes the PMDA page for the specific Japic Code to extract CTD, 添付文書, Patient Guide, IF, Siori, RMP, and RMP Materials."""
        import urllib.parse
        urls = {
            "ctd": [],
            "package_insert": [],
            "patient_guide": [],
            "if": [],
            "siori": [],
            "rmp": [],
            "rmp_material": [],
        }
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
                elif "rad-ar.or.jp" in href or "くすりのしおり" in text:
                    urls["siori"].append(full_url)
                elif "RMP" in href or "RMP" in text:
                    if "RMPm" in href or "資材" in text or "適正使用" in text or "ガイド" in text:
                        urls["rmp_material"].append(full_url)
                    else:
                        urls["rmp"].append(full_url)
                elif "ctd" in href.lower() or "common technical document" in text.lower() or "審査報告" in text or "報告書" in text:
                    urls["ctd"].append(full_url)
            
            # Fallback patterns if CTD not explicitly found
            if not urls["ctd"]:
                for a_tag in all_a:
                    href = a_tag["href"]
                    full_url = urllib.parse.urljoin(search_url, href)
                    if href.lower().endswith(".pdf"):
                        # If not already claimed by other categories
                        if not any(full_url in urls[cat] for cat in ["package_insert", "patient_guide", "if", "rmp", "rmp_material"]):
                            urls["ctd"].append(full_url)
                            break
        except Exception as e:
            print(f"[Warning] Failed to parse PMDA search url {search_url}: {e}")
        return urls

    def parse_radar_siori(self, japic_code: str, radar_url: str) -> str:
        """Scrapes Kusuri-no-Siori (くすりのしおり) page from RAD-AR site."""
        try:
            html = self.fetch_page(radar_url)
            soup = BeautifulSoup(html, "html.parser")
            
            # Look for Siori content block or print HTML link
            # RAD-AR typically has easy to parse layout, or text content in specific classes
            content_div = soup.find("div", class_="siori-content") or soup.find("div", id="siori_area")
            if content_div:
                return str(content_div)
            return html # fallback to raw html
        except Exception as e:
            print(f"[Warning] Failed to parse RAD-AR Siori url {radar_url}: {e}")
            return ""

    def download_file(self, url: str, dest_path: str):
        """Downloads file with rate-limiting."""
        if not url:
            return
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        time.sleep(2.0) # Rate limiting as required
        
        response = self.session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def run_auto_discovery(self) -> List[Dict[str, str]]:
        """Discovers Japic Codes and URLs automatically from PMDA category index URLs."""
        discovered_drugs = []
        for cat_url in self.auto_categories:
            try:
                html = self.fetch_page(cat_url)
                soup = BeautifulSoup(html, "html.parser")
                # Look for japic codes (usually 5 digit numbers or similar patterns)
                # PMDA list item structure search
                links = soup.find_all("a", href=True)
                for link in links:
                    href = link["href"]
                    text = link.get_text()
                    match = re.search(r"iyakuDetail/GeneralList/(\d+)", href)
                    if match:
                        code = match.group(1)
                        discovered_drugs.append({
                            "japic_code": code,
                            "name_ja": text.strip(),
                            "pmda_url": f"https://www.pmda.go.jp/PmdaSearch/iyakuDetail/GeneralList/{code}",
                            "radar_url": f"https://www.rad-ar.or.jp/kusuri/search/detail/{code}" # template URL
                        })
            except Exception as e:
                print(f"[Error] Failed during auto discovery on {cat_url}: {e}")
        return discovered_drugs[:5] # Limit auto-discovered list to 5 items for safety/demo purposes

    def collect(self) -> List[Dict[str, Any]]:
        """Main ingress collection loop. Downloads CTD, Package Insert, Patient Guide, IF, Siori, RMP, RMP Materials."""
        targets = []
        if self.mode in ["static", "both"]:
            targets.extend(self.static_drugs)
        if self.mode in ["auto", "both"]:
            targets.extend(self.run_auto_discovery())

        collected_results = []
        for drug in targets:
            code = drug["japic_code"]
            raw_dir = os.path.join(self.settings.paths.data_raw, f"JapicCode_{code}")
            os.makedirs(raw_dir, exist_ok=True)
            
            print(f"[Crawler] Processing Japic Code: {code}")
            
            # Crawl all PMDA detail URLs
            doc_urls = self.parse_pmda_details(code, drug["pmda_url"])
            
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
            for idx, url in enumerate(doc_urls["ctd"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"ctd{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["ctd_pdf"].append(dest)
                    print(f"  Downloaded CTD: {dest}")
                except Exception as e:
                    print(f"  [Error] Downloading CTD failed ({url}): {e}")
            
            # 2. Download 添付文書 (Package Insert)
            for idx, url in enumerate(doc_urls["package_insert"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"package_insert{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["package_insert"].append(dest)
                    print(f"  Downloaded Package Insert: {dest}")
                except Exception as e:
                    print(f"  [Error] Downloading Package Insert failed ({url}): {e}")
            
            # 3. Download 患者向医薬品ガイド (Patient Guide)
            for idx, url in enumerate(doc_urls["patient_guide"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"patient_guide{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["patient_guide"].append(dest)
                    print(f"  Downloaded Patient Guide: {dest}")
                except Exception as e:
                    print(f"  [Error] Downloading Patient Guide failed ({url}): {e}")
                    
            # 4. Download IF (Interview Form)
            for idx, url in enumerate(doc_urls["if"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"interview_form{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["if_pdf"].append(dest)
                    print(f"  Downloaded IF: {dest}")
                except Exception as e:
                    print(f"  [Error] Downloading IF failed ({url}): {e}")
                    
            # 5. Download くすりのしおり (Siori)
            siori_urls = doc_urls["siori"] if doc_urls["siori"] else [drug["radar_url"]]
            for idx, url in enumerate(siori_urls):
                suffix = f"_{idx+1}" if idx > 0 else ""
                siori_html = self.parse_radar_siori(code, url)
                if siori_html:
                    dest = os.path.join(raw_dir, f"siori{suffix}.html")
                    with open(dest, "w", encoding="utf-8") as f:
                        f.write(siori_html)
                    downloaded["siori_html"].append(dest)
                    print(f"  Saved Siori HTML: {dest}")
                else:
                    print(f"  [Warning] Failed to parse Siori HTML from {url}")
            
            # 6. Download RMP
            for idx, url in enumerate(doc_urls["rmp"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"rmp{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["rmp"].append(dest)
                    print(f"  Downloaded RMP: {dest}")
                except Exception as e:
                    print(f"  [Error] Downloading RMP failed ({url}): {e}")
                    
            # 7. Download RMP資材 (RMP Materials)
            for idx, url in enumerate(doc_urls["rmp_material"]):
                suffix = f"_{idx+1}" if idx > 0 else ""
                dest = os.path.join(raw_dir, f"rmp_material{suffix}.pdf")
                try:
                    self.download_file(url, dest)
                    downloaded["rmp_material"].append(dest)
                    print(f"  Downloaded RMP Material: {dest}")
                except Exception as e:
                    print(f"  [Error] Downloading RMP Material failed ({url}): {e}")
            
            collected_results.append({
                "japic_code": code,
                "name": drug.get("name_ja", "Unknown"),
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

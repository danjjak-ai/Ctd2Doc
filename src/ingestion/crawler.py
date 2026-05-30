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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.load_drug_list()

    def load_drug_list(self):
        with open(self.drug_list_path, "r", encoding="utf-8") as f:
            self.drug_config = yaml.safe_load(f)
        self.mode = self.drug_config.get("crawler", {}).get("mode", "both")
        self.static_drugs = self.drug_config.get("static_list", [])
        self.auto_categories = self.drug_config.get("crawler", {}).get("auto_categories", [])

    def fetch_page(self, url: str) -> str:
        """Fetches a page, automatically handling Japanese character encoding like Shift-JIS or EUC-JP."""
        response = requests.get(url, headers=self.headers, timeout=15)
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

    def parse_pmda_details(self, japic_code: str, search_url: str) -> Dict[str, str]:
        """Scrapes the PMDA page for the specific Japic Code to extract CTD (PDF) and IF (PDF) links."""
        urls = {"ctd": "", "if": ""}
        try:
            html = self.fetch_page(search_url)
            soup = BeautifulSoup(html, "html.parser")
            
            # Find links ending with .pdf containing CTD or Interview Form (IF) keywords
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                text = a_tag.get_text()
                
                # PMDA structure matching
                if "ctd" in href.lower() or "common technical document" in text.lower():
                    urls["ctd"] = href
                elif "if" in href.lower() or "interview" in text.lower() or "インタビューフォーム" in text:
                    urls["if"] = href
            
            # Fallback patterns if not explicitly found in text
            if not urls["ctd"]:
                pdf_links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".pdf")]
                for link in pdf_links:
                    if "ctd" in link.lower():
                        urls["ctd"] = link
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
        
        response = requests.get(url, headers=self.headers, stream=True, timeout=30)
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
        """Main ingress collection loop. Downloads CTD PDF, IF PDF, and Siori HTML for all targets."""
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
            
            # 1. PMDA details crawl for PDFs
            pmda_urls = self.parse_pmda_details(code, drug["pmda_url"])
            
            ctd_dest = os.path.join(raw_dir, "ctd.pdf")
            if pmda_urls["ctd"]:
                try:
                    self.download_file(pmda_urls["ctd"], ctd_dest)
                    print(f"  Downloaded CTD PDF: {ctd_dest}")
                except Exception as e:
                    print(f"  [Error] Downloading CTD failed: {e}")
                    ctd_dest = None
            else:
                ctd_dest = None

            if_dest = os.path.join(raw_dir, "interview_form.pdf")
            if pmda_urls["if"]:
                try:
                    self.download_file(pmda_urls["if"], if_dest)
                    print(f"  Downloaded IF PDF: {if_dest}")
                except Exception as e:
                    print(f"  [Error] Downloading IF failed: {e}")
                    if_dest = None
            else:
                if_dest = None
            
            # 2. RAD-AR crawl for Siori
            siori_html = self.parse_radar_siori(code, drug["radar_url"])
            siori_dest = os.path.join(raw_dir, "siori.html")
            if siori_html:
                with open(siori_dest, "w", encoding="utf-8") as f:
                    f.write(siori_html)
                print(f"  Saved Siori HTML: {siori_dest}")
            else:
                siori_dest = None
                
            collected_results.append({
                "japic_code": code,
                "name": drug.get("name_ja", "Unknown"),
                "ctd_pdf": ctd_dest,
                "if_pdf": if_dest,
                "siori_html": siori_dest
            })
            
        return collected_results

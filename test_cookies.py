import requests
from bs4 import BeautifulSoup
import chardet

session = requests.Session()
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
}
session.headers.update(headers)

# 1. Visit search main page to establish cookies
res1 = session.get("https://www.pmda.go.jp/PmdaSearch/iyakuSearch/")
print("Main page status:", res1.status_code)

# 2. Try accessing GeneralList page
res2 = session.get("https://www.pmda.go.jp/PmdaSearch/iyakuDetail/GeneralList/4291043")
print("Detail page status:", res2.status_code)

detected = chardet.detect(res2.content)
encoding = detected.get("encoding", "utf-8")
html = res2.content.decode(encoding, errors="ignore")
soup = BeautifulSoup(html, "html.parser")

title = soup.title.get_text().strip() if soup.title else "No title"
print("Title:", title)

# Check links
pdf_links = []
for a in soup.find_all("a", href=True):
    href = a['href']
    if ".pdf" in href.lower():
        pdf_links.append((a.get_text().strip(), href))
print(f"Found {len(pdf_links)} PDF links:")
for text, href in pdf_links:
    print(f"- {text}: {href}")

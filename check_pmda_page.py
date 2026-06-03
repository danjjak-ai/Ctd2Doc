import requests
from bs4 import BeautifulSoup
import chardet

url = "https://www.pmda.go.jp/PmdaSearch/iyakuDetail/GeneralList/520032615"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers, timeout=15)
detected = chardet.detect(response.content)
encoding = detected.get("encoding", "utf-8")
html = response.content.decode(encoding, errors="ignore")
soup = BeautifulSoup(html, "html.parser")

print("Status Code:", response.status_code)
print("HTML Snippet:")
print(html[:1000])



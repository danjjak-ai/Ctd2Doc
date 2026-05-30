# 구글 Colab Pro (L4 GPU) 기동 가이드

본 프로젝트는 **Colab Pro의 L4 GPU (24GB VRAM)** 환경에서 파인튜닝 및 추론, 평가 루프가 자동 구동되도록 최적화되어 있습니다. 아래 단계를 따라 Colab 노트북에서 파이프라인을 기동할 수 있습니다.

---

## 1. 필수 라이브러리 설치 (Colab 셀 실행)

Unsloth는 일반 PyPI 패키지보다 전용 wheel 링크를 통한 설치가 훨씬 안전하고 빠릅니다. 아래 명령어를 실행하십시오.

```bash
# 1. Unsloth 및 종속성 설치 (L4 GPU용 torch 2.2 / 2.3+ 호환 버전 자동 감지)
!pip install --no-deps "xformers<0.0.26" "assets/packaged_modules/" || true
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft whispers bitsandbytes datasets accelerate

# 2. 크롤러 및 대시보드 관련 패키지 설치
!pip install beautifulsoup4 chardet pyyaml chromadb sentence-transformers langchain-text-splitters streamlit plotly pynvml pyngrok
```

## 2. 환경 변수 및 설정 파일 마운트

1. 구글 드라이브 마운트 후 프로젝트 폴더로 이동합니다.
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Ctd2Doc  # 프로젝트 저장 경로
```

2. LLM-as-a-Judge API 평가를 위한 API 키 환경 변수 등록 (선택 사항 - API Fallback 동작 시 필요)
```python
import os
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"
```

## 3. 메인 자율 피드백 루프 기동

아래 코드를 Colab 셀에 넣어 실행하면 설정된 약품 리스트에 대해 자동으로 크롤링 → RAG 임베딩 → 생성 → 평가 → 피드백 학습(QLora) 루프가 동작합니다.

```python
from src.pipeline import AutonomousPipeline

# 파이프라인 인스턴스화 (config/settings.yaml 기반)
pipeline = AutonomousPipeline()

# 전체 약품 리스트 대상 루프 실행
# 85점 목표 도달 시까지 각 약품 단위(Batch Size = 1)로 학습
pipeline.run_all()
```

## 4. Streamlit 실시간 모니터링 대시보드 기동

Colab 백그라운드에서 Streamlit 서버를 띄우고, `pyngrok`을 통해 로컬 터널을 생성하여 브라우저에서 대시보드에 접근할 수 있습니다.

```python
# 1. ngrok Authtoken 등록
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")

# 2. Streamlit 대시보드 백그라운드 기동
import subprocess
subprocess.Popen(["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501"])

# 3. 로컬 터널 주소 출력
pub_url = ngrok.connect(8501)
print("대시보드 접속 URL:", pub_url.public_url)
```

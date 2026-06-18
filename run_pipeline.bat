@echo off
chcp 65001 > nul
REM Ctd2Doc 전체 프로세스 실행 배치 파일
REM Gemma-4-12B-IT 최적화 버전

echo ===================================================
echo [Ctd2Doc] Starting Entire Pipeline with Gemma-4-12B-IT
echo ===================================================

REM 1. 가상환경 확인 및 활성화
if not exist .venv (
    echo [.venv] Virtual environment not found! Creating one...
    uv venv
)
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b %errorlevel%
)

REM 2. 패키지 설치 확인
echo [Setup] Checking requirements...
uv pip install -e .
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install package.
    exit /b %errorlevel%
)

REM 3. 기본 모듈 자가 검증 실행
echo [Verify] Running test verification...
python -m pytest tests/test_gemma4_modules.py
if %errorlevel% neq 0 (
    echo [WARNING] Some tests did not pass. Please check your setup.
)

REM 4. 전체 파이프라인 dry-run 테스트
echo [Dry-run] Running dry-run validation...
python main.py --dry-run
if %errorlevel% neq 0 (
    echo [ERROR] Dry-run failed.
    exit /b %errorlevel%
)

REM 5. 실구동 설명 출력 및 선택 유도
echo ===================================================
echo [Ctd2Doc] Dry-run verification complete.
echo.
echo 실구동을 하려면 다음 명령어 중 하나를 수동으로 입력하세요:
echo  - 전체 의약품 실행: python main.py
echo  - 단일 의약품 실행: python main.py --single-drug [JapicCode]
echo  - 평가 리포트 조회: python main.py --status
echo ===================================================
pause

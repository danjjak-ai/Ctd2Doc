# Plan: Automated Self-Improving Pipeline Framework

## 1. 시스템 아키텍처 흐름도
[의약품 ID 리스트]
│
▼
[1. Crawler Agent] ──> PMDA/RAD-AR에서 약품별 CTD, IF, 시오리 다운로드
│
▼
[2. Data Parser] ───> PDF를 마크다운 변환 및 (CTD - 정답 문서) 페어 생성 및 격납
│
▼
[3. Vector Store] ──> CTD 문서를 Chunk링 후 임베딩 저장 (RAG 베이스 구축)
│
├──────────────────────────────┐
▼                              ▼
[4. Inference & Eval]          [5. Unsloth QLoRA Trainer]
모델이 RAG 기반 문서 생성            스코어 미달 시 자동 추가 학습 실행
│                              ▲
▼                              │
[6. Judge Agent] ────────────────────┘
정답 문서와 비교 후 스코어링 (목표치 도달 시 파이프라인 종료)


## 2. 상세 구현 단계

### 단계 1: 데이터 잉게스션 (Data Ingestion) 및 스토리지 격납
- 특정 의약품 성분명/코드 기반으로 스크래핑.
- 로컬 스토리지 구조 디렉토리 예시:
```text
  /data
    /raw
      /JapicCode_10023
        - ctd.pdf
        - interview_form.pdf
        - siori.html
    /processed
      /JapicCode_10023
        - ctd_cleaned.md
        - target_if.txt
        - target_siori.txt
단계 2: RAG 파이프라인 컴포넌트화
Embedding: intfloat/multilingual-e5-large를 활용하여 다국어(영/일) 벡터 스페이스 통합.

Vector DB: 가볍고 파일 기반 제어가 가능한 Chroma 또는 FAISS 사용.

Retriever: 입력 쿼리(예: "부작용 및 임상시험 결과")에 매칭되는 CTD 마크다운 청크 상위 5개 추출하여 LLM Context Window에 바인딩.

단계 3: 자율 피드백 평가 및 데이터 추가 루프 설계
Initial State: 학습되지 않은 순수 Gemma-2-9b-it 모델에 RAG를 붙여 샘플 약품 A, B의 결과를 생성함.

Evaluation: Judge Agent가 스코어 산출 (Target Score: 85점).

Trigger condition: 스코어가 85점 미만일 경우, 불일치 데이터와 오답 노트를 기반으로 학습 프롬프트(SFT Dataset Pair)를 생성하고 Unsloth 트레이너 기동.

Iteration: 다음 약품 C, D 데이터를 파이프라인에 주입하여 점수 점진적 상승 확인.

3. 핵심 자동화 알고리즘 (Pseudo Code)
Python
def autonomous_pipeline_loop(target_score=85):
    drug_list = ["Drug_A", "Drug_B", "Drug_C", "Drug_D"]
    active_model = "google/gemma-2-9b-it"
    
    for drug in drug_list:
        # 1. 문서 다운로드 및 전처리
        ctd_md, true_if, true_siori = prepare_data(drug)
        
        # 2. RAG 기반 생성 실행
        generated_if = run_rag_inference(active_model, ctd_md, template="interview_form")
        generated_siori = run_rag_inference(active_model, ctd_md, template="siori")
        
        # 3. Judge 에이전트 평가
        score, report = judge_agent.evaluate(generated_if, true_if, generated_siori, true_siori)
        print(f"[{drug}] Current Evaluation Score: {score}/100")
        
        if score >= target_score:
            print("목표 점수 달성. 다음 약품으로 이동.")
            continue
        else:
            print(f"점수 미달 ({score} < {target_score}). 추가 학습 데이터 생성 및 QLoRA 학습을 시작합니다.")
            # 실패 데이터셋 누적 격납
            save_to_sft_dataset(ctd_md, true_if, true_siori)
            
            # Unsloth 기반 자동 추가 학습 트리거
            active_model = run_unsloth_qlora_training(base_model=active_model, dataset_path="./sft_dataset")
# CTD2 Documentation

## 한국어

이 프로젝트는 화학 물질 데이터베이스(CTD)를 활용하여 질병과 화학 물질 간의 연관성을 탐색하고, 검색 및 라벨링 기능을 제공하는 파이썬 라이브러리입니다. 주요 기능은 다음과 같습니다:
- 벡터스토어를 이용한 효율적인 검색
- 질의에 대한 컨텍스트 생성
- 데이터셋 구축 및 학습 파이프라인

## English

This project is a Python library that leverages the Comparative Toxicogenomics Database (CTD) to explore associations between chemicals and diseases. It provides searching and labeling capabilities using a vector store. Key features include:
- Efficient retrieval with a vector store
- Context generation for queries
- Dataset building and training orchestration

## 日本語

このプロジェクトは、Comparative Toxicogenomics Database（CTD）を活用し、化学物質と病気の関連性を探索する Python ライブラリです。検索とラベリング機能をベクトルストアで提供します。主な機能は次のとおりです：
- ベクトルストアによる効率的な検索
- クエリのコンテキスト生成
- データセット構築と学習オーケストレーション

## Research
This project explores the integration of toxicogenomic data with modern LLM-based retrieval systems. By indexing the CTD, we aim to bridge the gap between chemical-disease interaction data and clinical decision support systems. Current research focuses on embedding space optimization for heterogeneous medical data.

## Plan
- Phase 1: Implement robust data ingestion pipelines for CTD raw files.
- Phase 2: Develop custom embedding models fine-tuned on toxicology terminology.
- Phase 3: Integrate RAG (Retrieval-Augmented Generation) patterns to improve chemical relationship extraction.
- Phase 4: Release a benchmarking suite for automated chemical labeling tasks.

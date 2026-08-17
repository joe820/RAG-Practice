# 🏢 Company Policy RAG API System
> **LangChain, ChromaDB, Google Gemini 3.6 Flash 기반의 사내규정 PDF 자동 인덱싱 및 AI 질의응답 REST API**

사내규정(취업규칙, 여비교통비규정, 복리후생규정 등) PDF 문서를 업로드하면 자동으로 텍스트를 청킹/임베딩하여 Vector DB에 적재하고, 임직원의 질문에 대해 명확한 출처(문서명 및 페이지)와 함께 신뢰도 높은 답변을 제공하는 RAG(Retrieval-Augmented Generation) 백엔드 시스템입니다.

---

## 🛠️ Tech Stack & Architecture

### Backend & AI Stack
* **Language / Framework:** Python 3.11+, FastAPI, Uvicorn
* **Orchestration:** LangChain (LCEL Pipeline)
* **LLM Engine:** Google Gemini API (`gemini-3.6-flash`, Temperature=0)
* **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (한국어/다국어 특화)
* **Vector Store:** ChromaDB (HNSW Indexing 기반 유사도 검색)
* **Document Parser:** PyPDF, RecursiveCharacterTextSplitter

### Architecture Flow
```text
[사내규정 PDF 업로드] ➡️ [PyPDF 파싱] ➡️ [Recursive Chunking (1000자 / Overlap 200자)]
     ⬇️
[Multilingual MiniLM 임베딩] ➡️ [Chroma Vector DB 적재 (HNSW Index)]
     ⬇️
[임직원 질의] ➡️ [Top-k 코사인 유사도 검색] ➡️ [Context Injection & Guardrail Prompt]
     ⬇️
[Gemini 3.6 Flash LLM 추론] ➡️ [출처(문서명, 페이지) 포함 정형화된 JSON 응답]
```

---

## 🎯 Key Engineering Challenges & Solutions

1. **Context 단절 및 수치 파편화 극복 (Chunking Optimization)**
   * **문제:** 초기 작은 청크 크기(500자)로 인해 사내규정 조항 제목과 세부 금액/기준 일수 등이 분리되어 검색 시 누락 발생.
   * **해결:** `chunk_size=1000`, `chunk_overlap=200`으로 확장하여 문맥의 연속성을 보장하고 검색 정확도 대폭 개선.

2. **한국어 의미 기반 검색 최적화 (Multilingual Embedding)**
   * 사내규정 한국어 텍스트 특성에 맞춰 다국어 전용 임베딩 모델(`paraphrase-multilingual-MiniLM-L12-v2`)을 도입하여 의미론적 코사인 유사도 검색 정밀도 향상.

3. **Hallucination(환각) 방지 Guardrail 구축**
   * System Prompt에 `Strict Context Constraint`를 적용하여, 제공된 사내규정에 없는 내용은 임의로 생성하지 않고 `제공된 사내규정 문서에서 관련 내용을 찾을 수 없습니다.`라고 명확히 거절하도록 통제.

4. **REST API 서비스 모듈화 및 라이프사이클 대응**
   * RAG 비즈니스 로직을 `RAGService` 싱글톤 클래스로 캡슐화하여 FastAPI 라우터와 관심사 분리.
   * LLM API의 Decommissioned/버전 변경 이슈에 유연하게 대응할 수 있도록 모델 인터페이스 추상화 및 최신 Gemini 플래그십 엔진 적용.

---

## 🚀 Quick Start

### 1. Repository Clone & Environment Setup
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/company-policy-rag.git
cd company-policy-rag

python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash 기준

pip install -r requirements.txt
```

### 2. Configure Environment Variables
`.env` 파일을 생성하고 Google AI Studio에서 발급받은 API Key를 등록합니다:
```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### 3. Run Server
```bash
uvicorn app.main:app --reload --port 8000
```
* **Swagger API Docs:** `http://localhost:8000/docs`

---

## 📡 API Specification

### 1. `POST /api/documents/upload`
사내규정 PDF 문서를 업로드하고 Vector DB에 청크 단위로 인덱싱합니다.
* **Request:** `multipart/form-data` (`file: UploadFile`)
* **Response:**
```json
{
  "status": "success",
  "filename": "취업규칙_2026.pdf",
  "chunks_created": 24,
  "message": "성공적으로 24개 청크가 Vector DB에 저장되었습니다."
}
```

### 2. `POST /api/chat`
사내규정 기반으로 질문을 던지고 출처가 포함된 답변을 수신합니다.
* **Request:**
```json
{
  "question": "경조사 휴가 일수와 지급 기준이 어떻게 되나요?"
}
```
* **Response:**
```json
{
  "question": "경조사 휴가 일수와 지급 기준이 어떻게 되나요?",
  "answer": "취업규칙 제32조에 따르면 본인 결혼 시 5일, 부모상 5일의 유급휴가가 부여됩니다...",
  "sources": [
    {
      "document": "취업규칙_2026.pdf",
      "page": "12"
    }
  ]
}
```

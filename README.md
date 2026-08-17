# 🏢 Company Policy RAG API System
> **LangChain, ChromaDB, Groq Llama 3.3 기반의 사내규정 PDF 자동 인덱싱 및 AI 질의응답 REST API**

사내규정(취업규칙, 여비교통비규정, 복리후생규정 등) PDF 문서를 업로드하면 자동으로 텍스트를 청킹/임베딩하여 Vector DB에 적재하고, 임직원의 질문에 대해 명확한 출처(문서명 및 페이지)와 함께 신뢰도 높은 답변을 제공하는 RAG(Retrieval-Augmented Generation) 백엔드 시스템입니다.

---

## 🛠️ Tech Stack & Architecture

### Backend & AI Stack
* **Language / Framework:** Python 3.11+, FastAPI, Uvicorn
* **Orchestration:** LangChain (LCEL Pipeline)
* **LLM Engine:** Groq API (`llama-3.3-70b-versatile`, Temperature=0)
* **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (다국어 지원)
* **Vector Store:** ChromaDB (HNSW Indexing 기반 유사도 검색)
* **Document Parser:** PyPDF, RecursiveCharacterTextSplitter

### Architecture Flow
```text
[PDF Upload] ➡️ [PyPDF Parsing] ➡️ [Recursive Chunking (1000자 / Overlap 200자)]
     ⬇️
[Multilingual Embedding] ➡️ [Chroma Vector DB Store (HNSW Index)]
     ⬇️
[User Query] ➡️ [Similarity Search (Top-k)] ➡️ [Context Injection & Guardrail Prompt]
     ⬇️
[Groq Llama 3.3 LLM] ➡️ [Structured Answer + Document & Page Citations]
```

---

## 🎯 Key Engineering Challenges & Solutions

1. **Context 단절 및 수치 파편화 극복 (Chunking Optimization)**
   * **문제:** 초기 작은 청크 크기(500자)로 인해 규정 조항 제목과 세부 내용/수치가 분리되어 검색 시 누락 발생.
   * **해결:** `chunk_size=1000`, `chunk_overlap=200`으로 확장하여 문맥의 연속성을 보장하고 검색 정확도 대폭 개선.

2. **한국어 의미 기반 검색 최적화 (Multilingual Embedding)**
   * 한국어 및 영어 도메인에 최적화된 다국어 임베딩 모델을 선정하여 코사인 유사도 검색 정확도 향상.

3. **Hallucination(환각) 방지 Guardrail 구축**
   * System Prompt에 `Strict Context Constraint`를 적용하여, 제공된 사내규정에 없는 내용은 임의로 생성하지 않고 모른다고 명시하도록 통제.

4. **REST API 서비스 모듈화**
   * RAG 비즈니스 로직을 `RAGService` 싱글톤 클래스로 캡슐화하고 FastAPI 라우터와 분리하여 유지보수성 확보.

---

## 🚀 Quick Start

### 1. Repository Clone & Environment Setup
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/company-policy-rag.git
cd company-policy-rag

python -m venv .venv
source .venv/Scripts/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
`.env` 파일을 생성하고 Groq API Key를 등록합니다:
```env
GROQ_API_KEY=your_groq_api_key_here
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

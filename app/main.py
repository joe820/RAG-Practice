import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.rag_service import RAGService

app = FastAPI(
    title="사내규정 AI 어시스턴트 API (Company Policy RAG)",
    description="사내규정 PDF 다중 업로드 및 Llama 3.3 기반 임직원 규정 질의응답 시스템",
    version="1.0.0"
)

rag_service = RAGService()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]

@app.get("/", summary="헬스체크 및 기본 안내")
def root():
    return {
        "message": "사내규정 AI 어시스턴트 API 서버가 정상 동작 중입니다.",
        "docs_url": "http://localhost:8000/docs"
    }

@app.post("/api/documents/upload", summary="사내규정 PDF 업로드 및 벡터화")
async def upload_document(file: UploadFile = File(...)):
    """사내규정 PDF 파일을 업로드하여 Vector DB에 저장합니다."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")
    
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        chunks_created = rag_service.ingest_pdf(temp_path, original_filename=file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": chunks_created,
            "message": f"성공적으로 {chunks_created}개 청크가 Vector DB에 저장되었습니다."
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/chat", response_model=ChatResponse, summary="사내규정 Q&A")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="질문 내용을 입력해주세요.")
    return rag_service.query(request.question)
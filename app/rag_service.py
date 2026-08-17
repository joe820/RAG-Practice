import os
import certifi
from dotenv import load_dotenv

# SSL 인증서 경로 강제 지정
os.environ['SSL_CERT_FILE'] = certifi.where()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAGService:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        # 다국어 임베딩 모델
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # Groq Llama 3.3 LLM
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        
        # Chroma Vector DB 로드/초기화
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def ingest_pdf(self, file_path: str, original_filename: str = None) -> int:
        """단일 사내규정 PDF를 파싱 및 청킹하여 Vector DB에 저장"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # 문서 메타데이터에 파일명 기록 (예: 취업규칙.pdf, 여비교통비규정.pdf)
        if original_filename:
            for doc in docs:
                doc.metadata['source'] = original_filename

        # 사내규정 조항 및 단락 보존을 위한 1000자 청크 / 200자 오버랩
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        if chunks:
            self.vectorstore.add_documents(chunks)
        return len(chunks)

    def query(self, question: str, k: int = 5) -> dict:
        """사내규정 검색 기반 질의응답"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        
        context_str = "\n\n".join([
            f"[규정 문서: {d.metadata.get('source', '사내규정')} - {d.metadata.get('page_label', d.metadata.get('page', 'N/A'))}페이지]\n{d.page_content}"
            for d in docs
        ])
        
        system_prompt = (
            "당신은 회사의 인사/총무/사내규정 안내 전문 어시스턴트입니다.\n"
            "아래 제공된 [사내규정 컨텍스트]만을 바탕으로 임직원의 질문에 명확하고 친절하게 답변하세요.\n"
            "규정에 명시되지 않은 내용은 임의로 추측하지 말고 '제공된 사내규정 문서에서 관련 내용을 찾을 수 없습니다.'라고 답변하세요.\n\n"
            "[사내규정 컨텍스트]:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "input": question})
        
        sources = [
            {
                "document": d.metadata.get("source", "사내규정"),
                "page": d.metadata.get("page_label", d.metadata.get("page", "N/A"))
            }
            for d in docs
        ]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
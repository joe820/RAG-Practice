import os
import certifi
from dotenv import load_dotenv

os.environ['SSL_CERT_FILE'] = certifi.where()
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGService:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Gemini 1.5 Flash (무료, 한국어 성능 우수, 빠른 속도)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )

    def ingest_pdf(self, file_path: str, original_filename: str = None) -> int:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        if original_filename:
            for doc in docs:
                doc.metadata['source'] = original_filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        if chunks:
            self.vectorstore.add_documents(chunks)
        return len(chunks)

    def query(self, question: str, k: int = 5) -> dict:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        
        context_str = "\n\n".join([
            f"[규정 문서: {d.metadata.get('source', '사내규정')} - {d.metadata.get('page_label', d.metadata.get('page', 1))}페이지]\n{d.page_content}"
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
                "document": str(d.metadata.get("source", "사내규정")),
                "page": str(d.metadata.get("page_label", d.metadata.get("page", "1")))
            }
            for d in docs
        ]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
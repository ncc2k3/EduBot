import os
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from logging import getLogger, basicConfig, INFO
import glob
from config.settings import *

# Cấu hình logging
basicConfig(level=INFO)
logger = getLogger("EduBot")

class EduBot:
    def __init__(self, file_path, vectorstore_dir):
        """
        Khởi tạo chatbot với file dữ liệu và vector store.
        """
        self.file_path = file_path
        self.vectorstore_dir = vectorstore_dir
        self.embedding_model = HuggingFaceEmbeddings(
            model_name= EMBEDDING_MODEL
        )
        self.vectorstore = None

    def create_or_load_vectorstore(self):
        """
        Kiểm tra và tải hoặc tạo vector store.
        """
        try:
            if os.path.exists(self.vectorstore_dir):
                logger.info(f"Vector store đã tồn tại: {self.vectorstore_dir}")
                self._load_vectorstore()
            else:
                self._create_vectorstore()
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc tải vector store: {e}")

    def _load_vectorstore(self):
        """
        Tải vector store từ thư mục đã lưu.
        """
        self.vectorstore = Chroma(
            persist_directory=self.vectorstore_dir,
            embedding_function=self.embedding_model
        )
        logger.info("Vector store đã được tải thành công.")

    def _create_vectorstore(self):
        """
        Tạo vector store từ file dữ liệu.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} không tồn tại.")

        logger.info("Đang tải và chia nhỏ tài liệu...")
        documents = self._load_and_split_docs()

        logger.info("Đang tạo vector store...")
        self.vectorstore = Chroma.from_documents(
            documents,
            self.embedding_model,
            persist_directory=self.vectorstore_dir
        )
        logger.info("Vector store đã được tạo và lưu thành công.")

    def _load_and_split_docs(self):
        """
        Đọc tất cả các file .txt và .md trong folder và chia nhỏ thành các đoạn.
        """
        documents = []
        try:
            # Lấy danh sách file trong folder
            file_paths = glob.glob(os.path.join(self.file_path, "*.txt")) + glob.glob(os.path.join(self.file_path, "*.md"))
            if not file_paths:
                raise FileNotFoundError("Không tìm thấy file nào trong thư mục được chỉ định.")

            for file_path in file_paths:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=800)
                chunks = splitter.split_text(content)
                logger.info(f"File {file_path} đã được chia thành {len(chunks)} đoạn.")
                documents.extend([Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks])
            
            return documents
        except Exception as e:
            logger.error(f"Lỗi khi đọc hoặc chia nhỏ tài liệu: {e}")
            raise

    def query_documents(self, query, top_k=3):
        """
        Truy vấn vector store để tìm tài liệu liên quan.
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vector store chưa được khởi tạo.")
            retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            results = retriever.get_relevant_documents(query)
            logger.info(f"Tìm thấy {len(results)} tài liệu liên quan.")
            return results
        except Exception as e:
            logger.error(f"Lỗi khi truy vấn tài liệu: {e}")
            return []

    def combine_context(self, documents):
        """
        Kết hợp nội dung từ các tài liệu để tạo ngữ cảnh.
        """
        try:
            combined_context = "\n\n".join([doc.page_content for doc in documents])
            logger.info("Đã kết hợp nội dung từ các tài liệu.")
            return combined_context
        except Exception as e:
            logger.error(f"Lỗi khi kết hợp ngữ cảnh: {e}")
            return ""
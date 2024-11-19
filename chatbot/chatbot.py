import os
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class StudentHandbookChatbot:
    def __init__(self, file_path, vectorstore_dir):
        """
        Khởi tạo chatbot với file dữ liệu và vector store.
        """
        self.file_path = file_path
        self.vectorstore_dir = vectorstore_dir
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = None

    def create_or_load_vectorstore(self):
        """
        Kiểm tra và tải hoặc tạo vector store.
        """
        if not os.path.exists(self.vectorstore_dir):
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File {self.file_path} không tồn tại.")
            
            print("Đang tải và chia nhỏ tài liệu...")
            documents = self._load_and_split_txt()

            print("Đang tạo vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding_function=self.embedding_model,
                persist_directory=self.vectorstore_dir
            )
        else:
            print(f"Vector store đã tồn tại tại: {self.vectorstore_dir}")
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_dir,
                embedding_function=self.embedding_model
            )

    def _load_and_split_txt(self):
        """
        Đọc file văn bản và chia nhỏ thành các đoạn.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(content)
        return [Document(page_content=chunk, metadata={"source": self.file_path}) for chunk in chunks]

    def query_documents(self, query, top_k=3):
        """
        Truy vấn vector store để tìm tài liệu liên quan.
        """
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        return retriever.invoke(query)

    def combine_context(self, documents):
        """
        Kết hợp nội dung từ các tài liệu để tạo ngữ cảnh.
        """
        return "\n\n".join([doc.page_content for doc in documents])

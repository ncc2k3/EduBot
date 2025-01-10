import os
import glob
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_core.tools import Tool
from config.settings import VECTORSTORE_DIR, DATA_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Tools")

def _load_and_split_docs():
    """
    Đọc tất cả các file .txt và .md trong folder và chia nhỏ thành các đoạn.
    """
    documents = []
    try:
        # Lấy danh sách file trong folder
        file_paths = glob.glob(os.path.join(DATA_DIR, "*.txt")) + glob.glob(os.path.join(DATA_DIR, "*.md"))
        if not file_paths:
            raise FileNotFoundError("Không tìm thấy file nào trong thư mục được chỉ định.")

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_text(content)
            logger.info(f"File {file_path} đã được chia thành {len(chunks)} đoạn.")
            documents.extend([Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks])
        
        print("Documents: ", documents)
        return documents
    except Exception as e:
        logger.error(f"Lỗi khi đọc hoặc chia nhỏ tài liệu: {e}")
        raise

def _create_vectorstore(embedding_model):
    """
    Tạo vector store từ file dữ liệu.
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"File {DATA_DIR} không tồn tại.")

    logger.info("Đang tải và chia nhỏ tài liệu...")
    documents = _load_and_split_docs()

    logger.info("Đang tạo vector store...")
    vectorstore = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=VECTORSTORE_DIR
    )
    logger.info("Vector store đã được tạo và lưu thành công.")
    
    return vectorstore

def _load_vectorstore(embedding_model):
    """
    Tải vector store từ thư mục đã lưu.
    """
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding_model
    )
    logger.info("Vector store đã được tải thành công.")
    return vectorstore

def create_or_load_vectorstore(embedding_model):
    """
    Kiểm tra và tải hoặc tạo vector store.
    """
    try:
        if os.path.exists(VECTORSTORE_DIR):
            logger.info(f"Vector store đã tồn tại: {VECTORSTORE_DIR}")
            return _load_vectorstore(embedding_model)
        else:
            return _create_vectorstore(embedding_model)
    except Exception as e:
        logger.error(f"Lỗi khi tạo hoặc tải vector store: {e}")
        raise

def create_chroma_retriever(embedding_model):
    """
    Tạo retriever từ vector store.
    """
    try:
        logger.info("Tạo retriever từ vector store...")
        vectorstore = create_or_load_vectorstore(embedding_model)
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        )
    except Exception as e:
        logger.error(f"Lỗi khi tạo retriever: {e}")
        raise
    
def handbook_tool(retriever):
    """
    Tạo công cụ để truy vấn sổ tay sinh viên.
    """
    try:
        logger.info("Handbook Tool được khởi tạo.")
        return Tool(
            name="Handbook Tool",
            func= lambda input, **kwargs: retriever.invoke(
                {"input": input, "chat_history": kwargs.get("chat_history", [])}
            ),
            description="Truy vấn thông tin từ sổ tay sinh viên"
        )
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo Handbook Tool: {e}")
        raise

def program_tool(retriever):
    """
    Tạo công cụ để truy vấn chương trình đào tạo.
    """
    try:
        logger.info("Program Tool được khởi tạo.")
        return Tool(
            name="Program Tool",
            func= lambda input, **kwargs: retriever.invoke(
                {"input": input, "chat_history": kwargs.get("chat_history", [])}
            ),
            description="Truy vấn thông tin từ chương trình đào tạo"
        )
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo Program Tool: {e}")
        raise

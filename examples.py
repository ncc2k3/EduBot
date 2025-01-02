import os
import logging
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Semantic Chunking and Retrieval")

# Cấu hình LLM
# LLM_MODEL = "llama3.1"
# llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

# Hàm load file .txt và .md
def load_documents(data_path):
    """
    Tải tài liệu từ thư mục dữ liệu. 
    """
    documents = []
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_name.endswith(".md"):
            loader = TextLoader(file_path, encoding="utf-8") # Ban đầu dùng UnstructuredMarkdownLoader nhưng nó bị mất ksi hiệu bảng
        else:
            continue
        documents.extend(loader.load())
    return documents

# Chunking dữ liệu bằng SemanticChunker
def chunk_documents(documents, embedding_model):
    """
    Sử dụng Semantic Chunker để chia nhỏ tài liệu dựa trên ngữ nghĩa.
    """
    semantic_chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    chunks = []
    for doc in documents:
        split_texts = semantic_chunker.create_documents([doc.page_content])
        for chunk in split_texts:
            chunks.append(Document(page_content=chunk.page_content, metadata=doc.metadata))
        
    return chunks

# Tạo vectorstore
def create_vectorstore(documents, vectorstore_dir, embedding_model):
    """
    Tạo vector store để lưu trữ các đoạn văn bản đã embedding.
    """
    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)
    logger.info("Đang tạo vector store...")
    vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=vectorstore_dir)
    return vectorstore

# Cấu hình đường dẫn
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "docs")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstores", "db_chroma_1")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
logger.info("Mô hình embedding được khởi tạo thành công.")

# Tạo vectorstore nếu chưa tồn tại
if not os.path.exists(VECTORSTORE_DIR):
    logger.info("Vector store không tồn tại. Đang tạo mới...")
    
    # Load documents
    logger.info("Đang tải tài liệu...")
    documents = load_documents(DATA_DIR)
    logger.info(f"Số lượng tài liệu: {len(documents)}")
    
    # Chunk documents
    logger.info("Đang chia nhỏ tài liệu...")
    chunked_documents = chunk_documents(documents, embedding_model)
    logger.info(f"Số lượng tài liệu sau khi chia nhỏ: {len(chunked_documents)}")
    
    # Create vectorstore
    logger.info("Đang tạo vector store...")
    vectorstore = create_vectorstore(chunked_documents, VECTORSTORE_DIR, embedding_model)
    logger.info("Vector store đã được tạo và lưu thành công.")
else:
    logger.info("Vector store đã tồn tại.")

# Load vectorstore
logger.info("Đang load vector store...")
vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embedding_model
)

# Khởi tạo retriever
logger.info("Khởi tạo retriever...")
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Truy vấn tài liệu
query = "ĐIỀU KIỆN TỐT NGHIỆP ngành Khoa Học máy tính"
documents = retriever.invoke(query)

for i, doc in enumerate(documents):
    print(f"Kết quả {i+1}:")
    print(doc.page_content)
    print("\n------\n")

# Tạo prompt
# Prompt trả lời câu hỏi
# Prompt này giúp AI hiểu rằng nó cần trả lời ngắn gọn và chính xác dựa trên ngữ cảnh được cung cấp
qa_system_prompt = (
    "Bạn là một trợ lý thông minh hỗ trợ sinh viên trường Đại Học Khoa Học Tự Nhiên, ĐHQG-HCM trả lời câu hỏi bằng Tiếng Việt."
    "Dựa vào các phần thông tin liên quan được cung cấp dưới đây, hãy trả lời câu hỏi một cách chính xác, ngắn gọn và chuyên nghiệp. "
    "Nếu phần thông tin cung cấp bị thừa hoặc không liên quan, hãy bỏ qua và trả lời câu hỏi một cách chính xác."
    "Câu trả lời cần đảm bảo ngắn gọn, chuyên nghiệp, chính xác, đúng trọng tâm và không chứa thông tin không liên quan. "
    "Nếu không có đủ thông tin trong cơ sở dữ liệu, hãy giải thích điều này và hướng dẫn sinh viên tìm kiếm ở nguồn khác phù hợp."
    "\n\n"
    "Thông tin được cung cấp: {context}"
)


# Tạo template prompt cho việc trả lời câu hỏi
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{query}"),
    ]
)

# Sử dụng LLM để sinh câu trả lời
logger.info("Đang tạo mô hình LLM...")
llm = ChatOllama(model="llama3.1", temperature=0.2)
logger.info("Mô hình LLM được khởi tạo thành công.")

logger.info("Đang tạo trả lời ...")
retrieved_context = "\n\n".join([doc.page_content for doc in documents])
formatted_prompt = qa_prompt.invoke({"context": retrieved_context, "query": query})
response = llm.invoke(formatted_prompt)

print("\nCâu trả lời từ AI:")
print(response)

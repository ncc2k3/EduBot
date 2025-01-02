import os
import logging
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Create a LLMs")

# Cấu hình LLM
LLM_MODEL = "llama3.1"  # Thay bằng model bạn đã cấu hình trong settings.py
llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

# HyDE Document Generation
template = """
Hãy viết một đoạn văn giống như trong bài báo khoa học để trả lời câu hỏi sau:
Câu hỏi: {question}
Đoạn văn:
Lưu ý: Đầu ra phải được viết bằng tiếng Việt với phong cách học thuật.
"""
prompt_hyde = ChatPromptTemplate.from_template(template)

# Tạo pipeline
generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser()
)


"""
tool:
    search_web

"""
# Thực thi với kiểm tra lỗi
try:
    question = "Các chuyên ngành của ngành Khoa học máy tính trường Đại học khoa học tự nhiên, ĐHQG-HCM"
    logger.info("Running HyDE query transformation...")
    response = generate_docs_for_retrieval.invoke({"question": question})
    logger.info("Generated passage successfully.")
    print(response)
except Exception as e:
    logger.error(f"Lỗi trong quá trình chuyển đổi truy vấn HyDE: {e}")


# Hàm kiểm tra bảng markdown
def is_markdown_table(content):
    logger.info("Kiểm tra bảng markdown...")
    """
    Kiểm tra nội dung có phải là bảng markdown hay không.
    """
    lines = content.strip().split("\n")
    if len(lines) < 2:  # Bảng markdown thường có ít nhất 2 dòng (header và separator)
        return False
    # Kiểm tra nếu dòng đầu hoặc dòng separator có nhiều dấu `|`
    return all("|" in line for line in lines[:2])

# Hàm load file .txt và .md
def load_documents(data_path):
    logger.info("Đang tải tài liệu...")
    documents = []
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_name.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path, encoding="utf-8")
        else:
            continue
        docs = loader.load()
        documents.extend(docs)
    return documents

# Chunking dữ liệu
def chunk_documents(documents, chunk_size=2000, chunk_overlap=500):
    logger.info("Đang chia nhỏ tài liệu...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        if is_markdown_table(doc.page_content):  # Kiểm tra bảng markdown
            # Nếu bảng lớn, chia thành từng phần nhỏ (giữ header)
            lines = doc.page_content.strip().split("\n")
            header = lines[0]
            body = lines[1:]

            for i in range(0, len(body), chunk_size):
                chunk_body = body[i:i + chunk_size]
                chunk_content = "\n".join([header] + chunk_body)
                chunks.append(Document(page_content=chunk_content, metadata=doc.metadata))
        else:
            split_texts = text_splitter.split_text(doc.page_content)
            for chunk in split_texts:
                chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunks

# Tạo vectorstore
def create_vectorstore(documents, vectorstore_dir, embedding_model):
    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)
    logger.info("Đang tạo vector store...")
    vectorstore = Chroma.from_documents(
        documents, 
        embedding_model, 
        persist_directory=vectorstore_dir)
    return vectorstore

# Cấu hình đường dẫn
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "docs")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstores","db_chroma_1")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Khởi tạo mô hình embedding
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
logger.info("Mô hình embedding được khởi tạo thành công.")

if not os.path.exists(VECTORSTORE_DIR):
    logger.info("Vector store không tồn tại. Đang tạo mới...")
    documents = load_documents(DATA_DIR)
    chunked_documents = chunk_documents(documents)
    vectorstore = create_vectorstore(chunked_documents, VECTORSTORE_DIR, embedding_model)
    logger.info("Vector store đã được tạo và lưu thành công.")
else:
    logger.info("Vector store đã tồn tại.")

# load vectorstore
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

query = "Các chuyên ngành trong ngành công nghệ thông tin"
document = retriever.invoke(query)

for i, doc in enumerate(document):
    print(f"Kết quả {i+1}:")
    print(doc.page_content)
    print("\n------\n")
    



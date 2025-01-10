import os

# Tên model embedding
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

"""
EMBEDDING_MODEL = "paraphase-multilingual-mpnet-base-v2"
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
"""

# Tên model ngôn ngữ lớn
LLM_MODEL = "qwen2.5:7b"

# Cấu hình đường dẫn
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "docs")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstores","db_chroma")

# Số lượng kết quả trả về từ retriever
TOP_K = 3

# Cấu hình chia nhỏ tài liệu
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 1000
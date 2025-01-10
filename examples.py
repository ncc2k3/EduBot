import os
import logging
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from typing import List
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Edubot")

""" ============== Functions ============== """
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

""" ======================================"""
""" ============== Cấu hình ============== """

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "docs")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstores", "db_chroma_1")
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = 'bkai-foundation-models/vietnamese-bi-encoder'

# Khởi tạo mô hình embedding
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
logger.info("Mô hình embedding được khởi tạo thành công.")

# Khởi tạo LLM 
llm = ChatOllama(model="qwen2.5:7b", temperature=0.0)
genai.configure()

model = genai.GenerativeModel("gemini-1.5-flash")
logger.info("Mô hình LLM được khởi tạo thành công.")

""" ========================================"""
""" ========================================"""

""" ============== Vectorstore ============== """
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
    search_type="mmr",
    search_kwargs={"k": 5}  
)

""" ==================== RAG-Fusion ==================== """
# Question
question = "Các chuyên ngành trong ngành công nghệ thông tin?"
# print("Câu hỏi: ", generate_queries.invoke({"question": question}))

from langchain.load import dumps, loads

### Reranking - Reciprocal Rank Fusion
def reciprocal_rank_fusion(results: list[list], k=100):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results
    
""" ============== Query Transformation ============== """
### Multiple queries
# Output parser will split the LLM result into a list of queries
from typing import List
from langchain.retrievers.multi_query import MultiQueryRetriever

# Parser
class ListOutputParser(BaseOutputParser[List[str]]):
    """Output parser to extract only numbered questions."""
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

# Prompt
query_generation_prompt_template = query_generation_prompt_template = query_generation_prompt_template = """
Bạn là trợ lý AI. Nhiệm vụ của bạn là tạo ra 5 phiên bản khác nhau của câu hỏi người dùng để tối ưu hóa tìm kiếm tài liệu từ cơ sở dữ liệu.

**Yêu cầu:**
- Chỉ trả về danh sách 5 câu hỏi biến thể, mỗi câu trên một dòng.
- Không cung cấp bất kỳ giải thích hoặc phân tích nào khác.
- Mỗi câu hỏi phải giữ nguyên ý nghĩa gốc nhưng có cách diễn đạt khác.

**Câu hỏi gốc:** {question}

**Đầu ra:**
- Danh sách gồm 5 câu hỏi biến thể, được định dạng như sau:
1. [Câu hỏi biến thể 1]
2. [Câu hỏi biến thể 2]
3. [Câu hỏi biến thể 3]
4. [Câu hỏi biến thể 4]
5. [Câu hỏi biến thể 5]

**Ví dụ 1:**
Câu hỏi gốc: "Các chuyên ngành của ngành công nghệ thông tin?"

Kết quả mong đợi:
1. Các chuyên ngành của ngành công nghệ thông tin?
2. Ngành công nghệ thông tin gồm những chuyên ngành nào?
3. Những chuyên ngành nào thuộc ngành công nghệ thông tin?
4. Ngành công nghệ thông tin có những lĩnh vực chuyên môn nào?
5. Các lĩnh vực nào nằm trong ngành công nghệ thông tin?

**Ví dụ 2:**
Câu hỏi gốc: "Các kỹ năng mà sinh viên ngành Trí tuệ nhân tạo cần đạt được sau khi tốt nghiệp là gì?"

Kết quả mong đợi:
1. Các kỹ năng mà sinh viên ngành Trí tuệ nhân tạo cần đạt được sau khi tốt nghiệp là gì?
2. Sinh viên ngành Trí tuệ nhân tạo cần đạt được những kỹ năng gì sau khi tốt nghiệp?
3. Những kỹ năng quan trọng nào sinh viên ngành Trí tuệ nhân tạo cần có khi tốt nghiệp?
4. Sau khi tốt nghiệp, sinh viên ngành Trí tuệ nhân tạo cần đạt những kỹ năng nào?
5. Các kỹ năng thiết yếu mà sinh viên ngành Trí tuệ nhân tạo cần sau khi tốt nghiệp là gì?
"""



QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template= query_generation_prompt_template
)

output_parser = ListOutputParser()

logger.info("Khởi tạo query generation chain ...")
query_generation_chain = QUERY_PROMPT | llm | output_parser

list_questions = query_generation_chain.invoke({"question": question})
print("Câu hỏi: ", list_questions)

# """ ============== Reranking  RAG-Fusion ============== """
retrieval_chain_rag_fusion = (
    query_generation_chain
    | retriever.map()  # Truy vấn từng câu hỏi trong danh sách
    | reciprocal_rank_fusion
)

docs = retrieval_chain_rag_fusion.invoke({"question": question})

# print("\n---\n")
# print("Kết quả sau khi tìm kiếm:")
print(docs[:5])  # In ra danh sách tài liệu sau re-ranking
print("\n---\n")
print(f"Số lượng tài liệu tìm được: {len(docs)}")

# Tạo prompt
# Prompt trả lời câu hỏi
# Prompt này giúp AI hiểu rằng nó cần trả lời ngắn gọn và chính xác dựa trên ngữ cảnh được cung cấp
qa_system_prompt = (
    "Bạn là trợ lý thông minh hỗ trợ sinh viên khoa Công nghệ thông tin của Đại học Khoa học Tự nhiên, Đại học Quốc gia TP.HCM trả lời câu hỏi bằng Tiếng Việt.\n"
    "Sử dụng thông tin từ cơ sở dữ liệu được cung cấp dưới đây để trả lời câu hỏi một cách chính xác và chuyên nghiệp.\n"
    "Tuân thủ các quy tắc sau:\n"
    "1. Nếu thông tin cung cấp đủ để trả lời:\n"
    "   - Tóm tắt nội dung chính của câu hỏi.\n"
    "   - Trả lời ngắn gọn và đầy đủ, trình bày từng ý mạch lạc.\n"
    "2. Nếu không đủ thông tin:\n"
    "   - Phản hồi rằng 'Tôi không đủ thông tin để trả lời câu hỏi này.'\n"
    "   - Gợi ý nguồn thông tin hoặc tài liệu để tìm hiểu thêm.\n"
    "3. Không bịa đặt thông tin hoặc đưa ra câu trả lời không có căn cứ.\n\n"
    "Dưới đây là ví dụ minh họa:\n"
    "Ví dụ 1:\n"
    "Câu hỏi: 'Các chuyên ngành trong ngành công nghệ thông tin?'\n"
    "Trả lời:\n"
    "   Các chuyên ngành của ngành Công nghệ thông tin là: \n"
    "   1. Mạng máy tính và Viễn thông\n "
    "   2. Công nghệ thông tin.\n\n"
    "Ví dụ 2:\n"
    "Câu hỏi: 'PGS.TS Lê Hoài Bắc giảng dạy những môn học nào?'\n"
    "Trả lời:\n"
    "   Không có thông tin về giảng viên này. Bạn có thể tìm hiểu thêm tại: https://www.fit.hcmus.edu.vn/\n\n"
    "Ví dụ 3:\n"
    "Câu hỏi: 'Các học phần liên quan đến môn Toán trong chương trình đào tạo?'\n"
    "Trả lời:\n"
    "   Dưới đây là các học phần liên quan đến môn Toán trong chương trình đào tạo:\n"
    "   - MTH00003 - Vi tích phân 1B\n"
    "   - MTH00081 - Thực hành Vi tích phân 1B\n"
    "   - MTH00004 - Vi tích phân 2B\n"
    "   - MTH00082 - Thực hành Vi tích phân 2B\n"
    "   - MTH00030 - Đại số tuyến tính\n"
    "   - MTH00083 - Thực hành Đại số tuyến tính\n"
    "   - MTH00040 - Xác suất thống kê \n"
    "   - MTH00085 - Thực hành Xác suất thống kê\n"
    "   - MTH00041 - Toán rời rạc\n"
    "   - MTH00086 - Thực hành Toán rời rạc\n"
    "   - MTH00050 - Toán học tổ hợp\n"
    "   - MTH00051 - Toán ứng dụng và thống kê (tự chọn)\n"
    "   - MTH00052 - Phương pháp tính (tự chọn)\n"
    "   - MTH00053 - Lý thuyết số (tự chọn)\n\n"
    "Thông tin được cung cấp: {context}\n\n"
    "Câu hỏi: {question}\n\n"
    "Trả lời:"
)


# Tạo template prompt cho việc trả lời câu hỏi
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{question}"),
    ]
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


model_gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

from operator import itemgetter
# Sử dụng LLM để sinh câu trả lời
logger.info("Khởi tạo final rag chain ...")
final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
    "question": itemgetter("question"),} 
    | qa_prompt
    | model_gemini
    | StrOutputParser()
)

logger.info("Đang tạo trả lời ...")
print(final_rag_chain.invoke({"question":question}))

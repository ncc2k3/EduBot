# import streamlit as st
from chatbot.tools import create_chroma_retriever
from langchain_ollama import ChatOllama
from config.settings import LLM_MODEL
import logging
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL
from chatbot.agent import create_agent_chains
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

# Khởi tạo ứng dụng Streamlit
# st.set_page_config(page_title="Chatbot Hỗ Trợ Sinh Viên", layout="wide")
# st.title("💬 EduBot ")

# Lịch sử hội thoại
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []

# Khởi tạo mô hình LLM
try:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7, language="vi")
    logger.info("Mô hình LLM được khởi tạo thành công.")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo mô hình LLM: {e}")
    # st.error("Không thể khởi tạo mô hình. Vui lòng kiểm tra lại cấu hình.")
    # st.stop()

# Khởi tạo embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info("Mô hình embedding được khởi tạo thành công.")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo embedding model: {e}")
    # st.error("Không thể khởi tạo embedding model. Vui lòng kiểm tra lại cấu hình.")
    # st.stop()
    
# Khởi tạo retriever
try:
    retriever = create_chroma_retriever(embedding_model)
    logger.info("Retriever được khởi tạo thành công.")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo retriever: {e}")
    # st.error("Không thể khởi tạo retriever. Vui lòng kiểm tra lại vectorstore.")
    # st.stop()


# Khởi tạo agent
try:
    agent_executor = create_agent_chains(llm, retriever)
    logger.info("Agent được khởi tạo thành công.")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo agent: {e}")
    # st.error("Không thể khởi tạo agent. Vui lòng kiểm tra lại cấu hình.")
    # st.stop()
    
chat_history = []

# query = "Các chuyên ngành trong ngành công nghệ thông tin"
query = "Các môn đại cương trong ngành công nghệ thông tin"
document = retriever.invoke(query)

for doc in document:
    print("\n\n ----- \n\n")
    print(doc.page_content)

# # Gửi câu hỏi đến agent và lấy kết quả
# response = agent_executor.invoke({"input": query, "chat_history": chat_history})
# print("AI:", response['output'])

# Giao diện nhập câu hỏi
# Hiển thị lịch sử hội thoại
# st.subheader("Lịch sử hội thoại")
# chat_placeholder = st.container()
# with chat_placeholder:
#     for chat in st.session_state["chat_history"]:
#         if chat["role"] == "human":
#             st.markdown(f"👤 **Bạn:** {chat['content']}")
#         elif chat["role"] == "assistant":
#             st.markdown(f"🤖 **Chatbot:** {chat['content']}")

# # Ô nhập câu hỏi ở bên dưới
# st.subheader("Nhập câu hỏi của bạn")
# query = st.text_input("", placeholder="Hãy nhập câu hỏi của bạn...")
# if query:
#     try:
#         # Gửi câu hỏi đến agent và lấy kết quả
#         response = agent_executor.invoke({"input": query, "chat_history": st.session_state["chat_history"]})
#         st.session_state["chat_history"].append({"role": "human", "content": query})
#         st.session_state["chat_history"].append({"role": "assistant", "content": response["output"]})

#         # Làm mới giao diện lịch sử hội thoại
#         chat_placeholder.empty()
#         with chat_placeholder:
#             for chat in st.session_state["chat_history"]:
#                 if chat["role"] == "human":
#                     st.markdown(f"👤 **Bạn:** {chat['content']}")
#                 elif chat["role"] == "assistant":
#                     st.markdown(f"🤖 **Chatbot:** {chat['content']}")

#     except Exception as e:
#         logger.error(f"Lỗi khi xử lý câu hỏi: {e}")
#         st.error("Lỗi trong quá trình xử lý câu hỏi. Vui lòng thử lại sau.")

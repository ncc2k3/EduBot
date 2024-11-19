import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from chatbot.chatbot import StudentHandbookChatbot
from chatbot.llm_handler import LLMHandler
from chatbot.prompt_templates import get_prompt_template
from config import *

# Khởi tạo chatbot và LLM handler
chatbot = StudentHandbookChatbot(file_path=FILE_PATH, vectorstore_dir=VECTORSTORE_DIR)
llm_handler = LLMHandler()
prompt_template = get_prompt_template()

# Load hoặc tạo vector store
chatbot.create_or_load_vectorstore()

# Thiết lập giao diện Streamlit
st.set_page_config(page_title="Chatbot - Sổ Tay Sinh Viên", layout="wide")
st.title("💬 Chatbot Hỏi Đáp - Sổ Tay Sinh Viên")

# Khởi tạo session state cho lịch sử hội thoại và trạng thái xử lý
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "waiting_for_response" not in st.session_state:
    st.session_state["waiting_for_response"] = False

# Hiển thị hội thoại
for msg in st.session_state["chat_history"]:
    if isinstance(msg, HumanMessage):
        st.chat_message("👤").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("🤖").write(msg.content)

# Nhập câu hỏi của người dùng
if not st.session_state.waiting_for_response:
    prompt = st.chat_input(placeholder="Nhập câu hỏi của bạn tại đây...")
    if prompt:
        # Thêm câu hỏi vào lịch sử
        user_message = HumanMessage(content=prompt)
        st.session_state["chat_history"].append(user_message)
        st.chat_message("👤").write(prompt)

        # Ngăn người dùng nhập tiếp trong khi đang xử lý
        st.session_state.waiting_for_response = True

        # Xử lý câu hỏi
        with st.spinner("Đang xử lý câu trả lời..."):
            documents = chatbot.query_documents(prompt)

            if not documents:
                answer = "❌ Không tìm thấy tài liệu phù hợp. Vui lòng thử lại với câu hỏi khác."
            else:
                # Tạo ngữ cảnh từ tài liệu
                context = chatbot.combine_context(documents)

                # Sinh câu trả lời từ template
                chat_history_text = "\n".join(
                    f"{'Người dùng' if isinstance(msg, HumanMessage) else 'Trợ lý'}: {msg.content}"
                    for msg in st.session_state["chat_history"]
                )
                answer = llm_handler.generate_answer(
                    context=context,
                    question=prompt,
                    chat_history=chat_history_text,
                    prompt_template=prompt_template,
                )

            # Thêm câu trả lời vào lịch sử
            ai_message = AIMessage(content=answer)
            st.session_state["chat_history"].append(ai_message)
            st.chat_message("🤖").write(answer)

        # Cho phép người dùng nhập câu hỏi mới
        st.session_state.waiting_for_response = False

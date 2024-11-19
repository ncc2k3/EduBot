from langchain.prompts import ChatPromptTemplate

def get_prompt_template():
    """
    Trả về template chuẩn cho chatbot.
    """
    return ChatPromptTemplate.from_template("""
        Bạn là một trợ lý thông minh hỗ trợ sinh viên về các câu hỏi liên quan đến sổ tay sinh viên.
        Dựa trên ngữ cảnh và lịch sử hội thoại bên dưới, trả lời câu hỏi của sinh viên một cách ngắn gọn, chính xác và chuyên nghiệp.

        Lịch sử hội thoại:
        {chat_history}

        Thông tin từ sổ tay sinh viên:
        {context}

        Câu hỏi hiện tại:
        {question}

        Câu trả lời của bạn:
    """)

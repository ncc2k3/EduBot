from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Prompt ngữ cảnh hóa câu hỏi
# Prompt này giúp AI hiểu rằng nó cần diễn đạt lại câu hỏi dựa trên lịch sử hội thoại
# để câu hỏi có thể tự hiểu mà không cần phụ thuộc vào ngữ cảnh trước đó
contextualize_q_system_prompt = (
    "Dựa vào lịch sử hội thoại và câu hỏi gần nhất của người dùng, "
    "nếu câu hỏi chứa các tham chiếu tới bối cảnh trong lịch sử hội thoại, "
    "hãy diễn đạt lại câu hỏi thành một câu hỏi độc lập có thể hiểu được "
    "mà không cần tham khảo lịch sử hội thoại. Không trả lời câu hỏi, "
    "chỉ cần diễn đạt lại nếu cần, nếu không thì giữ nguyên."
    "Lưu ý: Câu trả lời bằng Tiếng Việt."
)

# Tạo template prompt cho việc ngữ cảnh hóa câu hỏi
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Prompt trả lời câu hỏi
# Prompt này giúp AI hiểu rằng nó cần trả lời ngắn gọn và chính xác dựa trên ngữ cảnh được cung cấp
qa_system_prompt = (
    "Bạn là một trợ lý thông minh hỗ trợ trả lời câu hỏi bằng Tiếng Việt. Dựa vào các phần thông tin liên quan "
    "được cung cấp dưới đây, hãy trả lời câu hỏi một cách chính xác, ngắn gọn và chuyên nghiệp. "
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
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

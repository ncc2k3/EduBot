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
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
contextualize_q_system_prompt = """
    Bạn là một trợ lý AI thông minh, có khả năng hiểu ngữ cảnh từ lịch sử hội thoại để diễn đạt lại câu hỏi của người dùng thành một câu hỏi độc lập, dễ hiểu mà không cần tham chiếu đến lịch sử hội thoại. Hãy làm điều này theo cách rõ ràng, logic và ngắn gọn.

    **Hướng dẫn:**
    1. Đọc kỹ lịch sử hội thoại đã cung cấp.
    2. Xác định các tham chiếu bối cảnh trong câu hỏi của người dùng.
    3. Nếu câu hỏi chứa các tham chiếu không rõ ràng, diễn đạt lại câu hỏi thành một câu độc lập với đầy đủ thông tin ngữ cảnh từ lịch sử hội thoại.
    4. Nếu câu hỏi đã độc lập và rõ ràng, giữ nguyên.

    **Chuỗi tư duy (Chain of Thought - CoT):**
    - Xác định nội dung liên quan trong lịch sử hội thoại.
    - Kiểm tra xem câu hỏi của người dùng có thể hiểu được mà không cần bối cảnh hay không.
    - Nếu không, sử dụng thông tin từ lịch sử hội thoại để làm rõ câu hỏi.
    - Trả lời bằng câu hỏi đã được diễn đạt lại.

    **Đầu vào:**
    Lịch sử hội thoại: {chat_history}
    Câu hỏi gốc: {input}

    **Đầu ra:**
    Câu hỏi đã được diễn đạt lại: 

    **Ví dụ 1:**
    Lịch sử hội thoại:
    [
        {{"role": "user", "content": "Các chuyên ngành trong ngành công nghệ thông tin là gì?"}},
        {{"role": "assistant", "content": "Các chuyên ngành bao gồm: \n1. Mạng máy tính và Viễn thông\n2. Công nghệ phần mềm."}}
    ]
    Câu hỏi gốc: "Cần bao nhiêu tín chỉ để tốt nghiệp?"
    Chuỗi tư duy: 
    - Câu hỏi "Cần bao nhiêu tín chỉ để tốt nghiệp?" không rõ ngành học cụ thể.
    - Lấy thông tin về "Công nghệ thông tin" từ lịch sử hội thoại để làm rõ.
    Câu hỏi đã được diễn đạt lại: "Ngành Công nghệ thông tin cần bao nhiêu tín chỉ để tốt nghiệp?"

    **Ví dụ 2:**
    Lịch sử hội thoại:
    [
        {{"role": "user", "content": "Các chuyên ngành trong ngành công nghệ thông tin là gì?"}},
        {{"role": "assistant", "content": "Các chuyên ngành bao gồm: \n1. Mạng máy tính và Viễn thông\n2. Công nghệ phần mềm."}}
    ]
    Câu hỏi gốc: "Sinh viên năm 1, học kì 2 ngành Công nghệ thông tin cần học những môn gì?"
    Chuỗi tư duy: 
    - Câu hỏi đã rõ ràng và đầy đủ thông tin.
    - Không cần chỉnh sửa thêm.
    Câu hỏi đã được diễn đạt lại: "Sinh viên năm 1, học kì 2 ngành Công nghệ thông tin cần học những môn gì?"

    Lưu ý: Đảm bảo câu trả lời luôn bằng Tiếng Việt. Kết quả trả về chỉ hiện thị câu hỏi, không hiển thị gì thêm.
"""

contextualize_q_prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=contextualize_q_system_prompt
)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Thực hiện refine câu hỏi
refined_query_chain = (
    contextualize_q_prompt
    | llm  # Sử dụng Large Language Model để sinh output
    | StrOutputParser()  # Parser để chuyển đổi output về dạng string
)

chat_history =     [
        {"role": "user", "content": "Các chuyên ngành trong ngành công nghệ thông tin là gì?"},
        {"role": "assistant", "content": "Các chuyên ngành bao gồm: \n1. Mạng máy tính và Viễn thông\n2. Công nghệ phần mềm."}
    ]
    

user_input = "Cần bao nhiêu tín chỉ để tốt nghiệp?"
print(refined_query_chain.invoke({"chat_history": chat_history, "input": user_input}))
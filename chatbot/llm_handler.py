from langchain_community.chat_models import ChatOllama
from config.settings import *

class LLMHandler:
    def __init__(self, model_name=MODEL_NAME, temperature=0.7):
        """
        Khởi tạo mô hình LLM với các tham số.
        """
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def generate_answer(self, context, question, chat_history, prompt_template):
        """
        Sinh câu trả lời từ ngữ cảnh, câu hỏi và lịch sử hội thoại.
        """
        # Tạo prompt
        formatted_prompt = prompt_template.format_messages(
            context=context,
            question=question,
            chat_history=chat_history
        )

        # Gọi mô hình LLM
        response = self.llm.invoke(formatted_prompt[0].content)
        return response.content

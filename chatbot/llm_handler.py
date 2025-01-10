from langchain_community.chat_models import ChatOllama
from config.settings import *
from logging import getLogger, basicConfig, INFO
import time

# Cấu hình logging
basicConfig(level=INFO)
logger = getLogger("LLMHandler")

class LLMHandler:
    def __init__(self, model_name=MODEL_NAME, temperature=0.7):
        """
        Khởi tạo mô hình LLM với các tham số.
        """
        try:
            self.llm = ChatOllama(model=model_name, temperature=temperature)
            self.model_name = model_name
            self.temperature = temperature
            logger.info(f"Khởi tạo thành công mô hình: {model_name}")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo mô hình: {e}")
            raise

    def generate_answer(self, context, question, chat_history, prompt_template):
        """
        Sinh câu trả lời từ ngữ cảnh, câu hỏi và lịch sử hội thoại.
        """
        try:
            # Tạo prompt
            formatted_prompt = prompt_template.format_messages(
                chat_history=chat_history,
                context=context,
                question=question
            )

            # Kiểm tra độ dài của prompt
            if len(formatted_prompt[0].content) > 4096:  # Giả định mô hình có giới hạn 4096 token
                logger.warning("Prompt quá dài, cần rút gọn.")
                formatted_prompt[0].content = formatted_prompt[0].content[:4096]

            # Gọi mô hình LLM
            logger.info("Đang gửi yêu cầu đến mô hình...")
            start_time = time.time()
            response = self.llm.invoke(formatted_prompt[0].content)  # Hoặc .invoke nếu hỗ trợ
            elapsed_time = time.time() - start_time
            logger.info(f"Mô hình phản hồi trong {elapsed_time:.2f} giây.")

            return response.content

        except Exception as e:
            logger.error(f"Lỗi khi sinh câu trả lời: {e}")
            return "❌ Lỗi trong quá trình xử lý câu hỏi. Vui lòng thử lại sau."

    def switch_model(self, new_model_name, temperature=None):
        """
        Thay đổi mô hình LLM.
        """
        try:
            self.llm = ChatOllama(model=new_model_name, temperature=temperature or self.temperature)
            self.model_name = new_model_name
            logger.info(f"Đã chuyển sang mô hình mới: {new_model_name}")
        except Exception as e:
            logger.error(f"Lỗi khi thay đổi mô hình: {e}")
            raise

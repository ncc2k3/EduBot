from langchain.agents import create_react_agent, AgentExecutor
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_core.tools import Tool
import logging
from chatbot.prompts import contextualize_q_prompt, qa_prompt
from chatbot.tools import handbook_tool, program_tool

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Agents")

def create_agent_chains(llm, retriever):
    """
    Tạo agent sử dụng RAG chain với prompts tùy chỉnh.
    """
    try:
        # Tạo history-aware retriever
        logger.info("Khởi tạo history-aware retriever...")
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Tạo QA chain
        logger.info("Khởi tạo QA chain...")
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        # Tạo retrieval chain kết hợp history-aware retriever và QA chain
        logger.info("Kết hợp retriever và QA chain thành RAG chain...")
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Tạo tools dựa trên RAG chain
        logger.info("Tạo tools...")
        tools = [
            # handbook_tool(rag_chain),
            # program_tool(rag_chain)
            Tool(
                name="Program Tool",
                func= lambda input, **kwargs: retriever.invoke(
                    {"input": input, "chat_history": kwargs.get("chat_history", [])}
                ),
                description="Truy vấn thông tin"
            )
        ]

        # Pull react docstore prompt từ Hub
        logger.info("Tải react docstore prompt từ Hub...")
        react_docstore_prompt = hub.pull("hwchase17/react")
        # react_docstore_prompt = hub.pull("hwchase17/react-chat-json")

        # Tạo agent với prompt
        logger.info("Khởi tạo agent với RAG chain và tools...")
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=react_docstore_prompt
        )

        logger.info("Agent được khởi tạo thành công.")
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,  # Hiển thị thông tin chi tiết
            handle_parsing_errors=True  # Xử lý lỗi khi phân tích câu hỏi
        )

    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo agent: {e}")
        raise

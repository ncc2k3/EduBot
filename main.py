import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Define the directory containing the text file and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, 'vectorstores', 'db_chroma')

# Load HuggingFace Embeddings for multilingual support
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Initialize Chroma vector store
vector_db = Chroma(
    persist_directory=persistent_dir,
    embedding_function=embeddings
)

# Define the retriever for searching relevant documents
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
)

# # Example query from a student about the handbook
query = "Công thức tính điểm trung bình học kỳ là gì?"

# Retrieve the relevant documents based on the query
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
print("Number of documents retrieved:", len(relevant_docs))
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# Combine retrieved documents' content to form the context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Define LLM using Qwen2.5 model
llm = ChatOllama(model='qwen2.5:7b', temperature=0.7)

# Define prompt template for the chatbot using ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template("""
Bạn là một trợ lý thông minh cho sinh viên. Dựa trên thông tin được cung cấp bên dưới, hãy trả lời câu hỏi của sinh viên.

Thông tin từ sổ tay sinh viên: {context}

Câu hỏi: {question}

Câu trả lời của bạn:""")

# Format the prompt by passing in the context (retrieved documents) and question
final_prompt = prompt_template.format_messages(context=context, question=query)

# Get the response from the LLM
response = llm.invoke(final_prompt[0].content)  # final_prompt[0].content gives you the formatted prompt

###  -> 
# Display the final response
print("\n--- LLM Response ---")
print(response.content)

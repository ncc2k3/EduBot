# from sentence_transformers import SentenceTransformer
# from langchain_huggingface import HuggingFaceEmbeddings

# # INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
# sentences = ["Cô ấy là một người vui_tính .", "Cô ấy cười nói suốt cả ngày ."]

# model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

# # model = HuggingFaceEmbeddings(model_name='bkai-foundation-models/vietnamese-bi-encoder')
# print(model)


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

chain = llm | StrOutputParser()
print(chain.invoke(messages))

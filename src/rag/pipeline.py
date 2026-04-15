from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from .vector_store import build_vector_store

load_dotenv()

print('API KEY:', os.getenv('OPENAI_API_KEY'))

db = build_vector_store()

llm = ChatOpenAI(
    model="gpt-4o-mini",   # ✅ 正确
    api_key=os.getenv('OPENAI_API_KEY')
)

def ask(question: str):
    docs = db.similarity_search(question, k=3)

    context = '\n'.join([d.page_content for d in docs])

    prompt = f"""
你是一个助手，请基于以下内容回答问题：

{context}

问题：{question}
"""

    return llm.invoke(prompt)   # ✅ 正确
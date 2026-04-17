from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import logging

from .vector_store import build_vector_store

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("API 1111 KEY:", os.getenv("OPENAI_API_KEY"))

db = build_vector_store()

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus", 
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}
]


def ask(question: str):
    docs = db.similarity_search(question, k=3)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
你是一个助手，请基于以下内容回答问题：

{context}

要求：
    仅基于 {context} 回答问题，如果 {context} 中找不到对应答案，直接输出 '我尚未掌握该方面知识，等我学习之后再来探讨吧'

问题：{question}
"""
    logger.info("prompt: %s", prompt)
    return llm.invoke(prompt)

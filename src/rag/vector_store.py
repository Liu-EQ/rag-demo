from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import getpass

from .loader import load_docs

VECTOR_STORE_PATH = "src/data/vector_store"

print("embedding", os.getenv("OPENAI_API_KEY"))


# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


embeddings = HuggingFaceEmbeddings(
    model="Qwen/Qwen3-Embedding-0.6B",
)


def build_vector_store():
    if os.path.exists(VECTOR_STORE_PATH):
        print("Loading vector store from cache...")
        return FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )

    print("Creating new vector store...")
    text = str(load_docs())

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    # | splitter.split_text(text)
    docs = ['Python 是我目前在学习的语言'] 

    db = FAISS.from_texts(docs, embeddings)
    db.save_local(VECTOR_STORE_PATH)

    print("Vector store saved.")
    return db

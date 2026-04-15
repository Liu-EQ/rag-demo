from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .loader import load_docs

def build_vector_store():
    text = load_docs()

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    db = FAISS.from_texts(docs, embeddings)

    return db
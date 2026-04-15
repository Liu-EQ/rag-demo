from fastapi import FastAPI
from pydantic import BaseModel
from .rag.pipeline import ask


print('Hello World')

app = FastAPI()

class Query(BaseModel):
    question:str

@app.post('/ask')
def ask_api(query: Query):
    answer = ask(query.question)
    
    return {'answer: ', answer}
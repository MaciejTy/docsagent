from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Question(BaseModel):
    text: str
    max_results: int = 3


@app.get("/")
def read_root():
    return {"message": "Hello from DocsAgent"}


@app.post("/ask")
def ask_question(question: Question):
    return {
        "you_asked": question.text,
        "results_wanted": question.max_results,
    }
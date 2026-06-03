from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlmodel import SQLModel, create_engine, select, Field, Session

app = FastAPI()

engine = create_engine('sqlite:///docsagent.db')

class Question(BaseModel):
    text: str
    max_results: int = 3


class QuestionLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    max_results: int = 3
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

@app.on_event('startup')
def on_startup():
    SQLModel.metadata.create_all(engine)

@app.get("/")
def read_root():
    return {"message": "Hello from DocsAgent"}

@app.post("/ask")
def ask_question(question: Question):
    log = QuestionLog(text=question.text, max_results=question.max_results)
    with  Session(engine) as session:
        session.add(log)
        session.commit()
        session.refresh(log)
    return {
        "id": log.id,
        "text": question.text,
        "max_results": question.max_results,
        "created_at": log.created_at,
    }

@app.get("/questions")
def list_questions():
    with Session(engine) as session:
        questions = session.exec(select(QuestionLog)).all()
    return questions



@app.get("/questions/{question_id}")
def read_question(question_id: int):
    with Session(engine) as session:
        question = session.exec(
            select(QuestionLog).where(QuestionLog.id == question_id)
        ).first()
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return question

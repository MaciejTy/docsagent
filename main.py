from models import QuestionLog, Question
from fastapi import FastAPI, HTTPException
from sqlmodel import SQLModel, create_engine, select, Session
from contextlib import asynccontextmanager


engine = create_engine('sqlite:///docsagent.db')


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield

app = FastAPI(lifespan=lifespan)

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
        "text": log.text,
        "max_results": log.max_results,
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

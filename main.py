from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sqlmodel import Session, select

from db import create_db, engine
from embeddings import Embedder, embed, get_model
from models import AskResponse, Question, QuestionLog
from search import search


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db()
    get_model()  # ładujemy model raz, przy starcie, a nie przy pierwszym pytaniu
    yield

app = FastAPI(lifespan=lifespan)


def get_session():
    with Session(engine) as session:
        yield session


def get_embedder() -> Embedder:
    return embed


SessionDep = Annotated[Session, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


@app.get("/")
def read_root():
    return {"message": "Hello from DocsAgent"}


@app.post("/ask", response_model=AskResponse)
def ask_question(question: Question, session: SessionDep, embedder: EmbedderDep):
    log = QuestionLog(text=question.text, max_results=question.max_results)
    session.add(log)
    session.commit()
    session.refresh(log)

    query_vector = embedder([question.text])[0]
    results = search(session, query_vector, question.max_results)
    return AskResponse(question_id=log.id, text=log.text, results=results)


@app.get("/questions")
def list_questions(session: SessionDep):
    return session.exec(select(QuestionLog)).all()


@app.get("/questions/{question_id}")
def read_question(question_id: int, session: SessionDep):
    question = session.get(QuestionLog, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return question

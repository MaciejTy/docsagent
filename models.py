from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, StringConstraints
from pydantic import Field as PydanticField
from sqlmodel import Field, SQLModel


def utc_now():
    return datetime.now(timezone.utc)


class Question(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    max_results: int = PydanticField(default=3, ge=1, le=20)
    generate: bool = True


class QuestionLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    max_results: int = 3
    created_at: datetime = Field(default_factory=utc_now)


class Document(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    path: str = Field(unique=True, index=True)
    content_hash: str
    ingested_at: datetime = Field(default_factory=utc_now)


class Chunk(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    heading_path: str
    text: str
    embedding: bytes


class SearchResult(BaseModel):
    chunk_id: int
    source: str
    heading_path: str
    text: str
    score: float


class AskResponse(BaseModel):
    question_id: int
    text: str
    answer: str | None = None
    answer_error: str | None = None
    results: list[SearchResult]

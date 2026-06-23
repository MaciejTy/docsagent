from pydantic import BaseModel
from sqlmodel import SQLModel, Field
from datetime import datetime, timezone


class Question(BaseModel):
    text: str
    max_results: int = 3


class QuestionLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str
    max_results: int = 3
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
from sqlmodel import SQLModel, create_engine

import models  # noqa: F401 — rejestruje tabele w SQLModel.metadata

engine = create_engine("sqlite:///docsagent.db")


def create_db():
    SQLModel.metadata.create_all(engine)

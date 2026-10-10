import numpy as np
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import models  # noqa: F401 — rejestruje tabele
from embeddings import DIMENSION


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


class FakeEmbedder:
    """Deterministyczne wektory jednostkowe zamiast prawdziwego modelu."""

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        vectors = np.zeros((len(texts), DIMENSION), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, sum(map(ord, text)) % DIMENSION] = 1.0
        return vectors


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()

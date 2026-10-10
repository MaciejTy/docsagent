import numpy as np
import pytest
from pydantic import ValidationError
from sqlmodel import select

from embeddings import from_bytes, to_bytes
from models import Chunk, Document, Question


def test_question_rejects_blank_text():
    with pytest.raises(ValidationError):
        Question(text="   ")


def test_question_limits_max_results():
    assert Question(text="hi").max_results == 3
    with pytest.raises(ValidationError):
        Question(text="hi", max_results=0)
    with pytest.raises(ValidationError):
        Question(text="hi", max_results=21)


def test_vector_bytes_roundtrip():
    vector = np.array([0.1, -0.5, 2.0], dtype=np.float32)
    assert np.array_equal(from_bytes(to_bytes(vector)), vector)


def test_chunk_embedding_roundtrip_through_database(session):
    document = Document(path="a.md", content_hash="abc")
    session.add(document)
    session.commit()
    vector = np.array([0.5, -0.25], dtype=np.float32)
    session.add(Chunk(document_id=document.id, heading_path="A", text="t", embedding=to_bytes(vector)))
    session.commit()

    stored = session.exec(select(Chunk)).one()
    assert np.array_equal(from_bytes(stored.embedding), vector)

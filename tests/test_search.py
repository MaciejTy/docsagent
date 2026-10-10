import numpy as np
import pytest
from sqlmodel import select

from embeddings import to_bytes
from models import Chunk, Document
from search import search


def add_chunk(session, path, heading, text, vector):
    document = session.exec(select(Document).where(Document.path == path)).first()
    if document is None:
        document = Document(path=path, content_hash="x")
        session.add(document)
        session.flush()
    session.add(Chunk(
        document_id=document.id,
        heading_path=heading,
        text=text,
        embedding=to_bytes(np.array(vector, dtype=np.float32)),
    ))
    session.commit()


def query(*values):
    return np.array(values, dtype=np.float32)


def test_returns_best_matches_first(session):
    add_chunk(session, "a.md", "A", "about cats", [1, 0, 0])
    add_chunk(session, "a.md", "B", "about dogs", [0, 1, 0])
    add_chunk(session, "b.md", "C", "cats and dogs", [0.7071, 0.7071, 0])

    results = search(session, query(1, 0, 0), top_k=2)

    assert [r.text for r in results] == ["about cats", "cats and dogs"]
    assert [r.source for r in results] == ["a.md", "b.md"]
    assert [r.heading_path for r in results] == ["A", "C"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.7071, abs=1e-4)


def test_top_k_larger_than_index_returns_everything(session):
    add_chunk(session, "a.md", "A", "only one", [1, 0, 0])
    assert len(search(session, query(1, 0, 0), top_k=5)) == 1


def test_empty_index_returns_empty_list(session):
    assert search(session, query(1, 0, 0), top_k=3) == []

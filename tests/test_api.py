import pytest
from fastapi.testclient import TestClient

from ingestion import ingest_file
from main import app, get_embedder, get_session


@pytest.fixture
def client(session, fake_embedder):
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    # Bez "with": lifespan (ładowanie modelu, tworzenie pliku bazy) się nie uruchamia.
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ask_with_empty_index_returns_no_results(client):
    response = client.post("/ask", json={"text": "How do I run it?"})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "How do I run it?"
    assert isinstance(body["question_id"], int)
    assert body["results"] == []


def test_ask_returns_matching_chunks(client, session, tmp_path, fake_embedder):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nfirst\n\n## Install\n\nsecond\n", encoding="utf-8")
    ingest_file(session, path, fake_embedder)

    response = client.post("/ask", json={"text": "anything", "max_results": 1})

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert set(results[0]) == {"chunk_id", "source", "heading_path", "text", "score"}
    assert results[0]["source"] == path.as_posix()


@pytest.mark.parametrize("payload", [
    {},
    {"text": ""},
    {"text": "   "},
    {"text": "ok", "max_results": 0},
    {"text": "ok", "max_results": 21},
])
def test_ask_rejects_invalid_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_logs_question(client):
    question_id = client.post("/ask", json={"text": "logged?"}).json()["question_id"]

    response = client.get(f"/questions/{question_id}")

    assert response.status_code == 200
    assert response.json()["text"] == "logged?"


def test_unknown_question_returns_404(client):
    assert client.get("/questions/999").status_code == 404

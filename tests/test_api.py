import pytest
from fastapi.testclient import TestClient

from answering import AnswerError
from ingestion import ingest_file
from main import app, get_answerer, get_embedder, get_session


@pytest.fixture
def client(session, fake_embedder, fake_answerer):
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_answerer] = lambda: fake_answerer
    # Bez "with": lifespan (ładowanie modelu, tworzenie pliku bazy) się nie uruchamia.
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def indexed(session, tmp_path, fake_embedder):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nfirst\n\n## Install\n\nsecond\n", encoding="utf-8")
    ingest_file(session, path, fake_embedder)
    return path


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


def test_ask_returns_generated_answer(client, indexed, fake_answerer):
    response = client.post("/ask", json={"text": "anything", "max_results": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Generated answer [1]."
    assert body["answer_error"] is None
    assert len(body["results"]) == 2
    assert fake_answerer.calls == [("anything", 2)]


def test_generate_false_skips_answerer(client, indexed, fake_answerer):
    response = client.post("/ask", json={"text": "anything", "generate": False})

    body = response.json()
    assert body["answer"] is None
    assert body["answer_error"] is None
    assert len(body["results"]) == 2
    assert fake_answerer.calls == []


def test_empty_index_skips_answerer(client, fake_answerer):
    body = client.post("/ask", json={"text": "anything"}).json()

    assert body["answer"] is None
    assert body["answer_error"] is None
    assert body["results"] == []
    assert fake_answerer.calls == []


def test_answer_failure_still_returns_results(client, indexed, fake_answerer):
    fake_answerer.error = AnswerError("Claude credentials not found (set ANTHROPIC_API_KEY)")

    response = client.post("/ask", json={"text": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["answer_error"] == "Claude credentials not found (set ANTHROPIC_API_KEY)"
    assert len(body["results"]) == 2

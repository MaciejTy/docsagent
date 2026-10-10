# DocsAgent — etap 1 (retrieval) — plan implementacji

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **W tym projekcie:** Claude pisze kod i tłumaczy każdy krok; **commity i pushe robi użytkownik**. Kroki „Commit” to propozycja komend dla użytkownika — Claude ich nie uruchamia.

**Goal:** `POST /ask` zwraca najbardziej pasujące fragmenty zaindeksowanej dokumentacji markdown (źródło, ścieżka nagłówków, wynik podobieństwa).

**Architecture:** Skrypt `ingestion.py` tnie pliki `.md` po nagłówkach/akapitach, liczy embeddingi (MiniLM) i zapisuje je jako bajty w SQLite. `search.py` wczytuje wszystkie wektory do macierzy numpy i liczy podobieństwo cosinusowe (iloczyn skalarny znormalizowanych wektorów). FastAPI wstrzykuje sesję bazy i embedder przez `Depends`, co pozwala testom podmienić je na atrapy.

**Tech Stack:** Python 3.12, uv, FastAPI, SQLModel/SQLite, sentence-transformers (`all-MiniLM-L6-v2`), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-retrieval-design.md`

## Global Constraints

- Python ≥ 3.12; zależności przez `uv add` (nie pip).
- Model: `all-MiniLM-L6-v2`, wymiar 384, wektory float32 znormalizowane.
- Limit fragmentu: 1000 znaków; bloki kodu nigdy nie są dzielone.
- `max_results`: 1–20, domyślnie 3. `text`: niepusty po obcięciu białych znaków.
- Testy nigdy nie ładują prawdziwego modelu.
- Przykładowe dokumenty w `sample_docs/` (nie `docs/`).
- Uruchamianie: `uv run pytest`, `uv run python ingestion.py sample_docs/`, `uv run uvicorn main:app --reload`.

## Review Focus

1. Ponowne uruchomienie ingestion po edycji pliku → stare fragmenty znikają, brak duplikatów (Task 3: `test_changed_file_replaces_old_chunks`).
2. `#` w bloku kodu (komentarz bash/Python) → nie jest nagłówkiem (Task 1: `test_hash_inside_code_block_is_not_a_heading`).
3. Plik nie-UTF-8 w folderze → zgłoszony jako błąd, pozostałe pliki zaindeksowane (Task 3: `test_ingest_folder_reports_each_file_and_survives_bad_one`).
4. `max_results` większe niż liczba fragmentów → zwraca wszystkie, bez błędu (Task 4: `test_top_k_larger_than_index_returns_everything`).
5. Pytanie z samych spacji → 422, a nie pusty embedding (Task 2: `test_question_rejects_blank_text`, Task 5: `test_ask_rejects_invalid_input`).

---

### Task 0: Porządki w repo

**Co i dlaczego:** Sklejona linia w `.gitignore` sprawiła, że `.idea/` nie jest ignorowane. Przykładowa dokumentacja dostaje własny folder, którego skrypt nie pomyli ze specyfikacjami.

**Files:**
- Modify: `.gitignore` (linia `playground.py.idea/`)
- Modify: `pyproject.toml:4`
- Move: `docs_sample.md` → `sample_docs/fastapi.md`

- [ ] **Step 1: Napraw `.gitignore`** — zamień linię `playground.py.idea/` na `.idea/` (linia `playground.py` nad nią już istnieje).

- [ ] **Step 2: Przenieś przykładowy dokument**

```bash
mkdir -p sample_docs && mv docs_sample.md sample_docs/fastapi.md
```

- [ ] **Step 3: Opis projektu** — w `pyproject.toml`:

```toml
description = "Question answering over markdown documentation (semantic search API)"
```

- [ ] **Step 4: Commit (użytkownik)**

```bash
git rm -r --cached .idea
git add .gitignore pyproject.toml sample_docs/fastapi.md docs/superpowers
git commit -m "Clean up repo: untrack .idea, add sample docs and design spec"
```

---

### Task 1: Chunking markdowna

**Co i dlaczego:** Czysta funkcja tekst → fragmenty. Bez bazy i modelu, więc to najlepiej testowalna i najważniejsza dla jakości część.

**Files:**
- Create: `chunking.py`
- Create: `tests/__init__.py` (pusty), `tests/test_chunking.py`
- Modify: `pyproject.toml` (sekcja pytest)

**Interfaces:**
- Produces: `MAX_CHUNK_CHARS: int = 1000`; `@dataclass ChunkDraft(heading_path: str, text: str)` z metodą `embedding_input() -> str`; `chunk_markdown(text: str) -> list[ChunkDraft]`; `chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]`.

- [ ] **Step 1: Dodaj pytest**

```bash
uv add --dev pytest
```

Dopisz na końcu `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Napisz testy** — `tests/test_chunking.py`:

```python
from pathlib import Path

import pytest

from chunking import MAX_CHUNK_CHARS, ChunkDraft, chunk_markdown, chunk_text


def test_heading_path_follows_nesting():
    text = "# A\n\nintro\n\n## B\n\nbody b\n\n### C\n\nbody c\n\n## D\n\nbody d\n"
    chunks = chunk_markdown(text)
    assert [(c.heading_path, c.text) for c in chunks] == [
        ("A", "intro"),
        ("A > B", "body b"),
        ("A > B > C", "body c"),
        ("A > D", "body d"),
    ]


def test_hash_inside_code_block_is_not_a_heading():
    text = "# Setup\n\n```bash\n# install deps\npip install fastapi\n```\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Setup"
    assert "# install deps" in chunks[0].text


def test_cleans_front_matter_html_comments_and_anchors():
    text = (
        "---\ninclude_yaml:\n  sponsors: data/sponsors.yml\n---\n\n"
        "# FastAPI { #fastapi }\n\n<!-- sponsors -->\n"
        '<p align="center"><em>Fast to code</em></p>\n\n'
        "<style>\nh1 { display: none; }\n</style>\n"
    )
    assert chunk_markdown(text) == [ChunkDraft(heading_path="FastAPI", text="Fast to code")]


def test_long_section_is_split_on_paragraphs():
    paragraphs = [f"Paragraph {i} " + "x" * 400 for i in range(5)]
    chunks = chunk_markdown("# Long\n\n" + "\n\n".join(paragraphs))
    assert len(chunks) == 3
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    assert all(c.heading_path == "Long" for c in chunks)
    assert chunks[0].text.startswith("Paragraph 0")


def test_oversized_paragraph_falls_back_to_char_split():
    chunks = chunk_markdown("# Big\n\n" + "y" * 2500)
    assert len(chunks) == 3
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)


def test_code_block_is_never_split():
    code = "```python\n" + "\n\n".join(f"value_{i:03d} = {i}" for i in range(150)) + "\n```"
    chunks = chunk_markdown("# Code\n\n" + code + "\n")
    assert chunks == [ChunkDraft(heading_path="Code", text=code)]


def test_file_without_headings_has_empty_heading_path():
    chunks = chunk_markdown("just some text")
    assert chunks == [ChunkDraft(heading_path="", text="just some text")]
    assert chunks[0].embedding_input() == "just some text"


def test_embedding_input_prefixes_heading_path():
    assert ChunkDraft("A > B", "body").embedding_input() == "A > B\n\nbody"


def test_empty_and_markup_only_input_gives_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("# Title\n\n---\n\n<img src='x.png'>\n") == []


def test_sample_docs_are_chunked_sensibly():
    chunks = chunk_markdown(Path("sample_docs/fastapi.md").read_text(encoding="utf-8"))
    assert any(c.heading_path == "FastAPI > Example > Run it" for c in chunks)
    assert not any("<img" in c.text or "<a href" in c.text for c in chunks)


def test_chunk_text_does_not_emit_tail_made_only_of_overlap():
    assert chunk_text("a" * 1000, 1000, 100) == ["a" * 1000]


def test_chunk_text_empty_input():
    assert chunk_text("", 1000, 100) == []


def test_chunk_text_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)
```

- [ ] **Step 3: Uruchom — ma nie przejść**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chunking'`

- [ ] **Step 4: Implementacja** — `chunking.py`:

```python
import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 1000
FALLBACK_OVERLAP = 100

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
STYLE_OR_SCRIPT = re.compile(r"<(style|script)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
HEADING_ANCHOR = re.compile(r"\{\s*#[^}]*\}")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class ChunkDraft:
    heading_path: str
    text: str

    def embedding_input(self) -> str:
        if not self.heading_path:
            return self.text
        return f"{self.heading_path}\n\n{self.text}"


def chunk_markdown(text: str) -> list[ChunkDraft]:
    drafts = []
    for heading_path, lines in _split_sections(_strip_markup(text)):
        for chunk in _pack(_paragraphs(lines)):
            drafts.append(ChunkDraft(heading_path=heading_path, text=chunk))
    return drafts


def chunk_text(text, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start = start + (chunk_size - overlap)
    return chunks


def _strip_markup(text):
    text = FRONT_MATTER.sub("", text)
    text = HTML_COMMENT.sub("", text)
    return STYLE_OR_SCRIPT.sub("", text)


def _clean_heading(title):
    title = HEADING_ANCHOR.sub("", title)
    title = HTML_TAG.sub("", title)
    return title.replace("*", "").replace("`", "").strip()


def _path(stack):
    return " > ".join(title for _, title in stack)


def _split_sections(text):
    """Lista (heading_path, linie treści) — jedna pozycja na sekcję."""
    sections = []
    stack = []  # (poziom nagłówka, tytuł)
    lines = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            heading = HEADING.match(line)
            if heading:
                sections.append((_path(stack), lines))
                level = len(heading.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, _clean_heading(heading.group(2))))
                lines = []
                continue
            line = HTML_TAG.sub("", line)
        lines.append(line)
    sections.append((_path(stack), lines))
    return sections


def _paragraphs(lines):
    """Akapity rozdzielone pustą linią; blok kodu jest zawsze jednym akapitem."""
    paragraphs = []
    current = []
    in_fence = False
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            paragraphs.append("\n".join(current))
            current = []
        else:
            current.append(line)
    paragraphs.append("\n".join(current))
    return [p.strip() for p in paragraphs if re.search(r"\w", p)]


def _pack(paragraphs):
    """Skleja akapity w fragmenty nie dłuższe niż MAX_CHUNK_CHARS."""
    chunks = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > MAX_CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            if FENCE.match(paragraph):
                chunks.append(paragraph)
            else:
                chunks.extend(chunk_text(paragraph, MAX_CHUNK_CHARS, FALLBACK_OVERLAP))
        elif not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= MAX_CHUNK_CHARS:
            current += "\n\n" + paragraph
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks
```

- [ ] **Step 5: Uruchom — ma przejść**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: 13 passed

- [ ] **Step 6: Commit (użytkownik)**

```bash
git add chunking.py tests/ pyproject.toml uv.lock
git commit -m "Add markdown-aware chunking"
```

---

### Task 2: Modele danych, baza i serializacja wektorów

**Co i dlaczego:** Tabele `document` i `chunk`, walidacja pytania, wspólny silnik bazy dla API i skryptu, oraz funkcje embeddingów z leniwym ładowaniem modelu (import `sentence_transformers` trwa kilka sekund — nie chcemy go w testach).

**Files:**
- Modify: `models.py` (całość)
- Create: `db.py`, `embeddings.py`
- Create: `tests/conftest.py`, `tests/test_models.py`

**Interfaces:**
- Produces:
  - `models.Question(text: str, max_results: int = 3)`, `models.QuestionLog` (bez zmian w kolumnach), `models.Document(id, path, content_hash, ingested_at)`, `models.Chunk(id, document_id, heading_path, text, embedding: bytes)`, `models.SearchResult(chunk_id, source, heading_path, text, score: float)`, `models.AskResponse(question_id: int, text: str, results: list[SearchResult])`, `models.utc_now() -> datetime`.
  - `db.engine`, `db.create_db() -> None`.
  - `embeddings.DIMENSION = 384`, `embeddings.Embedder = Callable[[list[str]], np.ndarray]`, `embeddings.get_model()`, `embeddings.embed(texts: list[str]) -> np.ndarray` (kształt `(n, 384)`, float32, znormalizowane), `embeddings.to_bytes(vector) -> bytes`, `embeddings.from_bytes(data: bytes) -> np.ndarray`.
  - Fixtures w `tests/conftest.py`: `session` (SQLite w pamięci z tabelami), `fake_embedder` (obiekt wywoływalny jak `Embedder`, z licznikiem `calls`).

- [ ] **Step 1: Jawna zależność numpy** (używamy jej bezpośrednio, nie tylko przez sentence-transformers)

```bash
uv add numpy
```

- [ ] **Step 2: Fixtures** — `tests/conftest.py`:

```python
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
```

- [ ] **Step 3: Testy** — `tests/test_models.py`:

```python
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
```

- [ ] **Step 4: Uruchom — ma nie przejść**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'embeddings'`

- [ ] **Step 5: Implementacja** — `models.py` (cały plik):

```python
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
    results: list[SearchResult]
```

`db.py`:

```python
from sqlmodel import SQLModel, create_engine

import models  # noqa: F401 — rejestruje tabele w SQLModel.metadata

engine = create_engine("sqlite:///docsagent.db")


def create_db():
    SQLModel.metadata.create_all(engine)
```

`embeddings.py`:

```python
from collections.abc import Callable
from functools import lru_cache

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384

Embedder = Callable[[list[str]], np.ndarray]


@lru_cache(maxsize=1)
def get_model():
    # Import w środku: ładowanie torch trwa kilka sekund, a testy go nie potrzebują.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    vectors = get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def to_bytes(vector) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)
```

- [ ] **Step 6: Uruchom wszystkie testy**

Run: `uv run pytest -v`
Expected: 17 passed

- [ ] **Step 7: Commit (użytkownik)**

```bash
git add models.py db.py embeddings.py tests/conftest.py tests/test_models.py pyproject.toml uv.lock
git commit -m "Add document/chunk models, shared engine and embedding helpers"
```

> Uwaga: `main.py` w tym kroku nadal ma własny `engine` — przepinamy go w Task 5. Aplikacja działa bez zmian.

---

### Task 3: Ingestion

**Co i dlaczego:** Łączy chunking, embeddingi i bazę. Hash treści sprawia, że skrypt można uruchamiać wielokrotnie: niezmienione pliki są pomijane, zmienione — podmieniane w całości. Embeddingi liczone są *przed* zmianą bazy, więc błąd modelu nie zostawia połowicznego stanu.

**Files:**
- Modify: `ingestion.py` (całość — zastępuje obecny plik z samym `chunk_text`, który przeniósł się do `chunking.py`)
- Create: `tests/test_ingestion.py`

**Interfaces:**
- Consumes: `chunk_markdown`, `ChunkDraft.embedding_input()` (Task 1); `Document`, `Chunk`, `utc_now`, `create_db`, `engine`, `Embedder`, `embed`, `to_bytes` (Task 2).
- Produces: `ingest_file(session: Session, path: Path, embedder: Embedder) -> int | None` (liczba fragmentów albo `None` gdy plik niezmieniony); `ingest_folder(session: Session, folder: Path, embedder: Embedder) -> dict[str, str]` (ścieżka → `"indexed N chunks"` / `"unchanged"` / `"error: ..."`); `main(argv: list[str]) -> int`. Klucz dokumentu: `path.as_posix()`.

- [ ] **Step 1: Testy** — `tests/test_ingestion.py`:

```python
from sqlmodel import select

from embeddings import DIMENSION, from_bytes
from ingestion import ingest_file, ingest_folder, main
from models import Chunk, Document


def write(folder, name, content):
    path = folder / name
    path.write_text(content, encoding="utf-8")
    return path


def chunks_of(session, path):
    return session.exec(
        select(Chunk).join(Document).where(Document.path == path.as_posix())
    ).all()


def test_ingest_file_stores_chunks(session, tmp_path, fake_embedder):
    path = write(tmp_path, "guide.md", "# Guide\n\nfirst\n\n## Install\n\nsecond\n")

    assert ingest_file(session, path, fake_embedder) == 2

    chunks = chunks_of(session, path)
    assert [c.heading_path for c in chunks] == ["Guide", "Guide > Install"]
    assert [c.text for c in chunks] == ["first", "second"]
    assert len(from_bytes(chunks[0].embedding)) == DIMENSION


def test_unchanged_file_is_skipped(session, tmp_path, fake_embedder):
    path = write(tmp_path, "guide.md", "# Guide\n\nfirst\n\n## Install\n\nsecond\n")
    ingest_file(session, path, fake_embedder)

    assert ingest_file(session, path, fake_embedder) is None
    assert fake_embedder.calls == 1
    assert len(chunks_of(session, path)) == 2


def test_changed_file_replaces_old_chunks(session, tmp_path, fake_embedder):
    path = write(tmp_path, "guide.md", "# Guide\n\nfirst\n\n## Install\n\nsecond\n")
    ingest_file(session, path, fake_embedder)
    write(tmp_path, "guide.md", "# Guide\n\nonly one now\n")

    assert ingest_file(session, path, fake_embedder) == 1
    assert [c.text for c in chunks_of(session, path)] == ["only one now"]
    assert len(session.exec(select(Document)).all()) == 1


def test_ingest_folder_reports_each_file_and_survives_bad_one(session, tmp_path, fake_embedder):
    good = write(tmp_path, "good.md", "# Good\n\ntext\n")
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe not utf-8")
    (tmp_path / "nested").mkdir()
    deep = write(tmp_path / "nested", "deep.md", "# Deep\n\ntext\n")
    write(tmp_path, "notes.txt", "not markdown")

    report = ingest_folder(session, tmp_path, fake_embedder)

    assert len(report) == 3
    assert report[good.as_posix()] == "indexed 1 chunks"
    assert report[deep.as_posix()] == "indexed 1 chunks"
    assert report[bad.as_posix()].startswith("error:")


def test_main_rejects_missing_folder(tmp_path, capsys):
    assert main(["ingestion.py", str(tmp_path / "nope")]) == 1
    assert "not a directory" in capsys.readouterr().out
```

- [ ] **Step 2: Uruchom — ma nie przejść**

Run: `uv run pytest tests/test_ingestion.py -v`
Expected: FAIL — `ImportError: cannot import name 'ingest_file' from 'ingestion'`

- [ ] **Step 3: Implementacja** — `ingestion.py` (cały plik):

```python
import hashlib
import sys
from pathlib import Path

from sqlmodel import Session, delete, select

from chunking import chunk_markdown
from db import create_db, engine
from embeddings import Embedder, embed, to_bytes
from models import Chunk, Document, utc_now


def ingest_file(session: Session, path: Path, embedder: Embedder) -> int | None:
    content = path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    key = path.as_posix()

    document = session.exec(select(Document).where(Document.path == key)).first()
    if document is not None and document.content_hash == content_hash:
        return None

    drafts = chunk_markdown(content)
    vectors = embedder([d.embedding_input() for d in drafts]) if drafts else []

    if document is None:
        document = Document(path=key, content_hash=content_hash)
        session.add(document)
        session.flush()
    else:
        session.exec(delete(Chunk).where(Chunk.document_id == document.id))
        document.content_hash = content_hash
        document.ingested_at = utc_now()
        session.add(document)

    for draft, vector in zip(drafts, vectors):
        session.add(Chunk(
            document_id=document.id,
            heading_path=draft.heading_path,
            text=draft.text,
            embedding=to_bytes(vector),
        ))
    session.commit()
    return len(drafts)


def ingest_folder(session: Session, folder: Path, embedder: Embedder) -> dict[str, str]:
    report = {}
    for path in sorted(folder.rglob("*.md")):
        try:
            count = ingest_file(session, path, embedder)
        except (OSError, UnicodeDecodeError) as error:
            session.rollback()
            report[path.as_posix()] = f"error: {error}"
            continue
        report[path.as_posix()] = "unchanged" if count is None else f"indexed {count} chunks"
    return report


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python ingestion.py <folder>")
        return 2
    folder = Path(argv[1])
    if not folder.is_dir():
        print(f"error: {folder} is not a directory")
        return 1

    create_db()
    with Session(engine) as session:
        report = ingest_folder(session, folder, embed)
    for path, status in report.items():
        print(f"{path}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Uruchom wszystkie testy**

Run: `uv run pytest -v`
Expected: 22 passed

- [ ] **Step 5: Commit (użytkownik)**

```bash
git add ingestion.py tests/test_ingestion.py
git commit -m "Add idempotent markdown ingestion script"
```

---

### Task 4: Wyszukiwanie

**Co i dlaczego:** Wektory są znormalizowane, więc podobieństwo cosinusowe = iloczyn skalarny. Jedno mnożenie macierzy `(n, 384) @ (384,)` daje wyniki dla wszystkich fragmentów naraz.

**Files:**
- Create: `search.py`
- Create: `tests/test_search.py`

**Interfaces:**
- Consumes: `Chunk`, `Document`, `SearchResult` (Task 2); `from_bytes`, `to_bytes` (Task 2).
- Produces: `search(session: Session, query_vector: np.ndarray, top_k: int) -> list[SearchResult]` — posortowane malejąco po `score`; `[]` przy pustym indeksie.

- [ ] **Step 1: Testy** — `tests/test_search.py`:

```python
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
```

- [ ] **Step 2: Uruchom — ma nie przejść**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'search'`

- [ ] **Step 3: Implementacja** — `search.py`:

```python
import numpy as np
from sqlmodel import Session, select

from embeddings import from_bytes
from models import Chunk, Document, SearchResult


def search(session: Session, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
    rows = session.exec(select(Chunk, Document.path).join(Document)).all()
    if not rows:
        return []

    matrix = np.stack([from_bytes(chunk.embedding) for chunk, _ in rows])
    scores = matrix @ np.asarray(query_vector, dtype=np.float32)
    best = np.argsort(-scores, kind="stable")[:top_k]

    return [
        SearchResult(
            chunk_id=rows[i][0].id,
            source=rows[i][1],
            heading_path=rows[i][0].heading_path,
            text=rows[i][0].text,
            score=float(scores[i]),
        )
        for i in best
    ]
```

- [ ] **Step 4: Uruchom wszystkie testy**

Run: `uv run pytest -v`
Expected: 25 passed

- [ ] **Step 5: Commit (użytkownik)**

```bash
git add search.py tests/test_search.py
git commit -m "Add cosine similarity search over stored chunks"
```

---

### Task 5: `/ask` zwraca fragmenty

**Co i dlaczego:** Endpointy dostają sesję i embedder przez `Depends`. W produkcji to prawdziwa baza i model, w testach — baza w pamięci i atrapa. Model ładuje się raz, przy starcie serwera.

**Files:**
- Modify: `main.py` (całość)
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `create_db`, `engine` (Task 2); `Embedder`, `embed`, `get_model` (Task 2); `Question`, `QuestionLog`, `AskResponse` (Task 2); `search` (Task 4); `ingest_file` (Task 3, tylko w teście).
- Produces: `main.app`, `main.get_session`, `main.get_embedder` (do nadpisywania w testach).

- [ ] **Step 1: Testy** — `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Uruchom — ma nie przejść**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_embedder' from 'main'`

- [ ] **Step 3: Implementacja** — `main.py` (cały plik):

```python
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
```

- [ ] **Step 4: Uruchom wszystkie testy**

Run: `uv run pytest -v`
Expected: 34 passed

- [ ] **Step 5: Commit (użytkownik)**

```bash
git add main.py tests/test_api.py
git commit -m "Return matching documentation chunks from /ask"
```

---

### Task 6: Sprawdzenie na prawdziwym modelu + README

**Co i dlaczego:** Testy używają atrapy — tu sprawdzamy, że prawdziwy model faktycznie znajduje właściwe sekcje.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Zaindeksuj przykładową dokumentację**

Run: `uv run python ingestion.py sample_docs/`
Expected: `sample_docs/fastapi.md: indexed N chunks` (N > 0). Drugie uruchomienie: `sample_docs/fastapi.md: unchanged`.

- [ ] **Step 2: Uruchom serwer i zadaj pytanie**

```bash
uv run uvicorn main:app --reload
# w drugim terminalu:
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"text": "How do I run the development server?"}' | python -m json.tool
```

Expected: wśród 3 wyników jest fragment z `heading_path` zawierającym `Run it`; `score` malejąco.

- [ ] **Step 3: README** — `README.md`:

````markdown
# DocsAgent

API do zadawania pytań o dokumentację w markdownie. Dokumenty są cięte na
fragmenty według nagłówków, zamieniane na wektory (`all-MiniLM-L6-v2`)
i przeszukiwane semantycznie.

## Uruchomienie

```bash
uv sync
uv run python ingestion.py sample_docs/    # indeksowanie (ponowne uruchomienie pomija niezmienione pliki)
uv run uvicorn main:app --reload           # API na http://localhost:8000, dokumentacja: /docs
```

## API

- `POST /ask` — `{"text": "...", "max_results": 3}` → najlepiej pasujące fragmenty
  (`source`, `heading_path`, `text`, `score`).
- `GET /questions`, `GET /questions/{id}` — historia pytań.

## Testy

```bash
uv run pytest
```

## Ograniczenia

- Usunięcie pliku z folderu nie usuwa jego fragmentów z bazy.
- Wyszukiwanie liczy podobieństwo do wszystkich fragmentów — wystarcza do kilkudziesięciu tysięcy.
````

- [ ] **Step 4: Commit (użytkownik)**

```bash
git add README.md
git commit -m "Add README"
```

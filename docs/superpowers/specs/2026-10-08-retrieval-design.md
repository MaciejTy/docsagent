# DocsAgent — etap 1: wyszukiwanie w dokumentacji (retrieval)

Data: 2026-10-08

## Cel

DocsAgent to API, do którego indeksuje się dokumentację w markdownie, a następnie
zadaje pytania przez `POST /ask`. W etapie 1 API zwraca najbardziej pasujące
fragmenty dokumentacji wraz ze źródłem i wynikiem podobieństwa. Etap 2 (osobna
specyfikacja) dołoży odpowiedź generowaną przez LLM na podstawie tych fragmentów.

Projekt ma pozostać prosty i czytelny: stack bez zmian (FastAPI, SQLModel/SQLite,
sentence-transformers), bez zewnętrznych baz wektorowych.

## Decyzje

| Temat | Decyzja | Uzasadnienie |
|---|---|---|
| Wynik `/ask` | Etap 1: fragmenty; etap 2: + odpowiedź LLM | Wyszukiwanie to fundament, testowalny w izolacji |
| Dodawanie dokumentów | Skrypt CLI na folderze `.md` | Indeksowanie oddzielone od serwowania; `ingest_file()` gotowe do użycia przez przyszły endpoint |
| Przechowywanie wektorów | SQLite (bajty float32) + numpy | Skala: setki–tysiące fragmentów; brute force trwa milisekundy |
| Chunking | Po nagłówkach markdown, dalej po akapitach, limit ~1000 znaków | Wykorzystuje strukturę dokumentu; mieści się w limicie ~256 tokenów modelu |
| Model | `all-MiniLM-L6-v2` (384 wymiary) | Już używany w prototypie, lokalny, darmowy |

## Moduły

| Plik | Odpowiedzialność | Zależy od |
|---|---|---|
| `models.py` | `Question`, `QuestionLog`, `Document`, `Chunk`, `SearchResult` | — |
| `db.py` | Silnik bazy, tworzenie tabel | `models` |
| `chunking.py` | `chunk_markdown(text) -> list[ChunkDraft]` — czysta funkcja | — |
| `embeddings.py` | Jednorazowe ładowanie modelu; `embed(texts) -> np.ndarray` (znormalizowane) | sentence-transformers |
| `ingestion.py` | `ingest_file(session, path)`, `ingest_folder(session, path)`; uruchamialny jako skrypt | `chunking`, `embeddings`, `db` |
| `search.py` | `search(session, query_vector, top_k) -> list[SearchResult]` | `db`, numpy |
| `main.py` | Endpointy | wszystkie powyższe |

`search()` przyjmuje gotowy wektor pytania (nie tekst), dzięki czemu testuje się go
na ręcznie wpisanych wektorach bez ładowania modelu.

## Model danych

- **`document`**: `id`, `path` (unikalny), `content_hash` (sha256 treści), `ingested_at`.
- **`chunk`**: `id`, `document_id` (FK), `heading_path` (np. `FastAPI > Example > Run it`),
  `text`, `embedding` (bajty, 384 × float32).
- **`questionlog`**: bez zmian.

## Chunking — reguły

1. Usuń komentarze HTML, tagi HTML i front matter YAML (`---` na początku pliku).
   Usuń znaczniki kotwic `{ #id }` z nagłówków.
2. Podziel na sekcje po liniach nagłówków `#`–`######`. Linie wewnątrz bloków kodu
   (```` ``` ````) nie są nagłówkami.
3. Każda sekcja dostaje `heading_path` — ścieżkę nagłówków od najwyższego poziomu.
4. Sekcja ≤ 1000 znaków → jeden fragment. Dłuższa → dzielona po akapitach
   (pusta linia), łączonych zachłannie do limitu. Blok kodu jest traktowany jak jeden
   akapit i nie jest dzielony. Pojedynczy akapit tekstowy > 1000 znaków → `chunk_text`
   (podział po znakach z zakładką) jako awaryjny fallback.
5. Fragmenty puste po oczyszczeniu są pomijane.
6. Do embeddingu idzie `heading_path + "\n\n" + text`; w bazie i w odpowiedzi
   `text` jest przechowywany bez ścieżki.

## Ingestion

- `uv run python ingestion.py sample_docs/` — indeksuje wszystkie `*.md` rekurencyjnie.
- Plik z niezmienionym `content_hash` jest pomijany. Zmieniony: stare fragmenty
  usuwane, nowe wstawiane, w jednej transakcji.
- Błąd odczytu jednego pliku jest raportowany; skrypt kontynuuje.
- Wyjście: jedna linia na plik (`indexed N chunks` / `unchanged` / `error: ...`).

## API

**`POST /ask`** — żądanie: `{"text": str (niepusty), "max_results": int (1–20, domyślnie 3)}`.
Odpowiedź:

```json
{
  "question_id": 12,
  "text": "How do I run the development server?",
  "results": [
    {"chunk_id": 41, "source": "sample_docs/fastapi.md",
     "heading_path": "FastAPI > Example > Run it", "text": "...", "score": 0.71}
  ]
}
```

- Pytanie jest zawsze logowane do `questionlog`.
- Pusty indeks → `200`, `"results": []`.
- Złe dane wejściowe → `422` (walidacja Pydantic).
- Model ładowany raz w `lifespan`.

`GET /`, `GET /questions`, `GET /questions/{id}` — bez zmian.

## Testy (pytest)

- `test_chunking.py` — ścieżki nagłówków, `#` w bloku kodu, podział długiej sekcji,
  usuwanie HTML/front matter/kotwic, pusty plik, fallback dla długiego akapitu.
- `test_search.py` — ranking i `top_k` na ręcznych wektorach; pusty indeks.
- `test_ingestion.py` — tymczasowa baza; pominięcie niezmienionego pliku,
  podmiana fragmentów zmienionego; atrapa embeddera.
- `test_api.py` — `TestClient`; kształt odpowiedzi `/ask`, 422, pusty indeks;
  atrapa embeddera.

Żaden test nie pobiera prawdziwego modelu.

## Porządki (krok 0)

- `.gitignore`: rozdzielić linię `playground.py.idea/`; `git rm -r --cached .idea`.
- `docs_sample.md` → `sample_docs/fastapi.md` (nie do `docs/`, bo tam leżą specyfikacje, które skrypt by zaindeksował).
- Opis w `pyproject.toml`.
- README na końcu etapu.

## Poza zakresem etapu 1

Upload przez API, usuwanie dokumentów przez API, próg minimalnego podobieństwa,
generowanie odpowiedzi przez LLM (etap 2), baza wektorowa.

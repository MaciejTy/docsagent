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

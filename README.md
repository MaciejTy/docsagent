# DocsAgent

API do zadawania pytań o dokumentację w markdownie. Dokumenty są cięte na
fragmenty według nagłówków, zamieniane na wektory (`all-MiniLM-L6-v2`)
i przeszukiwane semantycznie; na podstawie znalezionych fragmentów Claude
pisze odpowiedź ze wskazaniem źródeł.

## Uruchomienie

```bash
uv sync
uv run python ingestion.py sample_docs/    # indeksowanie (ponowne uruchomienie pomija niezmienione pliki)
uv run uvicorn main:app --reload           # API na http://localhost:8000, dokumentacja: /docs
```

## API

- `POST /ask` — `{"text": "...", "max_results": 3, "generate": true}` → odpowiedź
  napisana przez Claude na podstawie znalezionych fragmentów (`answer`, z odwołaniami
  `[n]` do pozycji w `results`) oraz same fragmenty (`source`, `heading_path`, `text`,
  `score`). `"generate": false` zwraca tylko fragmenty. Gdy odpowiedzi nie udało się
  wygenerować, `answer` to `null`, a `answer_error` mówi dlaczego — fragmenty są
  zwracane mimo to.
- `GET /questions`, `GET /questions/{id}` — historia pytań.

## Konfiguracja

| Zmienna | Domyślnie | Znaczenie |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Klucz API Claude; bez niego działa samo wyszukiwanie |
| `DOCSAGENT_MODEL` | `claude-opus-5-5` | Model generujący odpowiedzi (np. `claude-haiku-5-5`, tańszy) |

## Testy

```bash
uv run pytest
```

## Ograniczenia

- Usunięcie pliku z folderu nie usuwa jego fragmentów z bazy.
- Wyszukiwanie liczy podobieństwo do wszystkich fragmentów — wystarcza do kilkudziesięciu tysięcy.

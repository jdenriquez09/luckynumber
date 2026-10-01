# Project Hermes

PCSO Lotto Statistical Analysis Platform.

## Stack
- **Backend:** Python 3.12, DuckDB (`data/processed/lotto.duckdb`), FastAPI (`api/`)
- **Dashboard:** Streamlit (`dashboard/app.py`) — 10 pages
- **Scraper:** curl-cffi async scraper (`src/lotto/scrape.py`) — bypasses Cloudflare
- **AI:** Azure OpenAI via `src/lotto/azure_ai.py` (sync AzureOpenAI client, o1/o3 model — no system role, no temperature, use `max_completion_tokens`)

## Key conventions
- DuckDB single-writer: open a fresh connection per query, close in `finally`
- `draws` table PK: `(draw_date, game, draw_time)`
- Number columns are cast to `Int64` (nullable) in `load_draws()` — never use `.values.flatten().astype(float)` directly; use `pd.array(..., dtype=object)` then filter `None`/`pd.NA`
- All analysis functions return `AnalysisResult(name, table, conclusion, meta)`
- Azure deployment name: `SOL` — loaded from `.env` at project root

## Running the dashboard
```
cd C:\luckynumber
streamlit run dashboard/app.py
```

## Running the API
```
cd C:\luckynumber
uvicorn api.main:app --reload
```

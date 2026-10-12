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

## Choosing numbers: ONE standard way
- Every ticket is half ML scores, half lucky numbers of the draw date (3 + 3 on 6-number games).
  Built only by `src/lotto/tickets.py::generate_tickets` (ML half: `lean_ml.py`; lucky half:
  `lucky.py`, which also takes the user's own numbers and optional AI-suggested numbers).
- Tickets are made in ONE place: the **Make Tickets** page (and API `/analysis/{game}/probability`,
  default `method=hybrid`). The AI produces the lucky numbers: for the draw date it proposes a shortlist
  (`lucky.ask_ai_lucky_numbers`, from calendar facts computed in code) and the lucky half of every ticket is
  drawn from it, ranked by the date rules (rules alone if AI is off or fails; API opt-in `ai_lucky=true`).
  The AI also explains tickets. It never edits a ticket. The Arena can only backtest the rules-only version.
- **Strategy Arena** is the scoreboard: it measures the standard recipe (`hybrid`) against baselines
  (uniform, hot, overdue, ML only, lucky only, crowd-avoiding). It does not make tickets.
- The Make Tickets page is one numbered process: 1 choose the game, 2 set up lucky numbers,
  3 get the tickets (save with 🎯), 4 read why (AI explanation), 5 score saved bets against the real
  draw (this feeds the ML half). The heat map and On This Day are background, not steps.
- Do not add another ticket generator to a page. Add it as an Arena strategy (`arena.STRATEGY_LABELS`),
  measure it, and promote it into `generate_tickets` only if the numbers justify it.
- ML-assisted and Uniform random remain on the Probability page only as baselines under "Advanced".

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

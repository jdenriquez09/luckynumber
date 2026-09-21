"""FastAPI application entry point."""
import asyncio
import sys
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.db import get_connection
from api.routers import games, draws, analysis
from src.lotto.games import GAMES, get_game
from src.lotto.scrape import scrape_pick6
from src.lotto.clean import load_and_clean

app = FastAPI(
    title="PCSO Lotto Analysis API",
    version="1.0.0",
    description="Statistical analysis of PCSO draw results. No predictions made.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:8501",   # Streamlit
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(games.router)
app.include_router(draws.router)
app.include_router(analysis.router)


@app.on_event("startup")
def startup() -> None:
    conn = get_connection()
    conn.close()


@app.get("/")
def root() -> dict:
    return {
        "service": "PCSO Lotto Analysis API",
        "docs": "/docs",
        "games": "/games",
        "draws": "/draws",
        "analysis": "/analysis",
    }


async def _scrape_task(game_id: str, years: list[int]) -> None:
    game_cfg = GAMES[game_id]
    conn = get_connection()
    raw_rows = await scrape_pick6(game_cfg, years, db_conn=conn)
    inserted, flagged = load_and_clean(raw_rows, game_cfg, conn)
    conn.close()
    import logging
    logging.getLogger(__name__).info(
        "scrape complete game=%s inserted=%d flagged=%d", game_id, inserted, flagged
    )


@app.post("/scrape/{game_id}")
async def trigger_scrape(
    game_id: str,
    years: list[int] | None = None,
    background_tasks: BackgroundTasks = None,  # type: ignore[assignment]
) -> dict:
    game_cfg = GAMES.get(game_id.lower())
    if game_cfg is None:
        raise HTTPException(404, f"Game '{game_id}' not found")
    if game_cfg.ordered:
        raise HTTPException(400, "Scraping digit games not yet implemented")

    from datetime import datetime
    current_year = datetime.now().year
    target_years = years or list(range(2021, current_year + 1))

    background_tasks.add_task(_scrape_task, game_cfg.id, target_years)
    return {"status": "started", "game": game_cfg.id, "years": target_years}

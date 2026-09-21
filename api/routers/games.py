from fastapi import APIRouter
from src.lotto.games import GAMES, GameConfig

router = APIRouter(prefix="/games", tags=["games"])


@router.get("/")
def list_games() -> list[dict]:
    return [
        {
            "id": g.id,
            "name": g.name,
            "pool": g.pool,
            "picks": g.picks,
            "draw_days": list(g.draw_days),
            "draw_times": list(g.draw_times),
            "ordered": g.ordered,
        }
        for g in GAMES.values()
    ]


@router.get("/{game_id}")
def get_game(game_id: str) -> dict:
    g = GAMES.get(game_id.lower())
    if g is None:
        from fastapi import HTTPException
        raise HTTPException(404, f"Game '{game_id}' not found")
    return {
        "id": g.id,
        "name": g.name,
        "pool": g.pool,
        "picks": g.picks,
        "draw_days": list(g.draw_days),
        "draw_times": list(g.draw_times),
        "ordered": g.ordered,
        "low_boundary": g.low_boundary,
    }

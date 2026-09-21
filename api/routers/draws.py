from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from api.db import get_connection

router = APIRouter(prefix="/draws", tags=["draws"])


@router.get("/")
def get_draws(
    game: Optional[str] = Query(None),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    conn = get_connection(read_only=True)
    conditions = []
    params: list = []

    if game:
        conditions.append("game = ?")
        params.append(game.lower())
    if from_date:
        conditions.append("draw_date >= ?")
        params.append(from_date)
    if to_date:
        conditions.append("draw_date <= ?")
        params.append(to_date)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    count_row = conn.execute(f"SELECT COUNT(*) FROM draws {where}", params).fetchone()
    total = count_row[0] if count_row else 0

    params_page = params + [limit, offset]
    rows = conn.execute(
        f"SELECT * FROM draws {where} ORDER BY draw_date DESC LIMIT ? OFFSET ?",
        params_page,
    ).fetchdf()

    conn.close()
    return {"total": total, "limit": limit, "offset": offset, "data": rows.to_dict(orient="records")}


@router.get("/coverage")
def coverage() -> list[dict]:
    conn = get_connection(read_only=True)
    rows = conn.execute("""
        SELECT game, COUNT(*) AS draws,
               MIN(draw_date)::VARCHAR AS first_draw,
               MAX(draw_date)::VARCHAR AS last_draw
        FROM draws GROUP BY game ORDER BY game
    """).fetchdf()
    conn.close()
    return rows.to_dict(orient="records")

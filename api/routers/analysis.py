import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from api.db import get_connection
from src.lotto.games import GAMES
import src.lotto.analysis as an
import src.lotto.predict as pred

router = APIRouter(prefix="/analysis", tags=["analysis"])

_PICK6_TESTS = {
    "freq_chi_square": an.freq_chi_square,
    "distribution_tests": an.distribution_tests,
    "pair_cooccurrence": an.pair_cooccurrence,
    "gap_analysis": an.gap_analysis,
    "sequence_tests": an.sequence_tests,
    "monte_carlo": an.monte_carlo,
    "jackpot_dynamics": an.jackpot_dynamics,
    "date_effects": an.date_effects,
}

_DIGIT_TESTS = {
    "digit_uniformity": an.digit_uniformity,
    "digit_repeat_rate": an.digit_repeat_rate,
}


@router.get("/tests")
def list_tests() -> dict:
    return {"pick6": list(_PICK6_TESTS.keys()), "digit": list(_DIGIT_TESTS.keys())}


@router.get("/{game_id}/{test_name}")
def run_analysis(game_id: str, test_name: str) -> dict:
    game_cfg = GAMES.get(game_id.lower())
    if game_cfg is None:
        raise HTTPException(404, f"Game '{game_id}' not found")

    if game_cfg.ordered:
        func = _DIGIT_TESTS.get(test_name)
    else:
        func = _PICK6_TESTS.get(test_name)

    if func is None:
        raise HTTPException(404, f"Test '{test_name}' not available for game '{game_id}'")

    conn = get_connection(read_only=True)
    num_cols = ", ".join(f"n{i}" for i in range(1, game_cfg.picks + 1))
    df = conn.execute(f"""
        SELECT draw_date, weekday, game, draw_time,
               pub_1, pub_2, pub_3, pub_4, pub_5, pub_6,
               {num_cols}, jackpot_php, notes
        FROM draws WHERE game = ?
        ORDER BY draw_date
    """, [game_cfg.id]).fetchdf()
    conn.close()

    if df.empty:
        raise HTTPException(404, f"No draw data for game '{game_id}'")

    try:
        result = func(df, game_cfg)
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {
        "game": game_cfg.id,
        "test": result.name,
        "conclusion": result.conclusion,
        "table": result.table.to_dict(orient="records"),
        "meta": result.meta,
    }


def _load_draws(game_cfg) -> "pd.DataFrame":
    import pandas as pd
    conn = get_connection(read_only=True)
    num_cols = ", ".join(f"n{i}" for i in range(1, game_cfg.picks + 1))
    df = conn.execute(f"""
        SELECT draw_date, weekday, game, draw_time,
               pub_1, pub_2, pub_3, pub_4, pub_5, pub_6,
               {num_cols}, jackpot_php, notes
        FROM draws WHERE game = ?
        ORDER BY draw_date
    """, [game_cfg.id]).fetchdf()
    conn.close()
    return df


@router.get("/{game_id}/probability")
def get_probability(
    game_id: str,
    window: int | None = Query(default=None, description="Limit to last N draws; omit for all"),
    n_suggestions: int = Query(default=5, ge=1, le=20),
    draw_time: str | None = Query(default=None, description="Filter to a specific draw time, e.g. '14:00'"),
) -> dict:
    """Return per-number heat map scores and weighted-random combination suggestions."""
    game_cfg = GAMES.get(game_id.lower())
    if game_cfg is None:
        raise HTTPException(404, f"Game '{game_id}' not found")

    df = _load_draws(game_cfg)
    if df.empty:
        raise HTTPException(404, f"No draw data for game '{game_id}'")

    heat = pred.number_heat_map(df, game_cfg, window=window, draw_time=draw_time)
    combos = pred.suggest_combinations(df, game_cfg, n=n_suggestions, window=window, draw_time=draw_time)
    summary = pred.heat_map_summary(heat)

    total = int((df["draw_time"] == draw_time).sum()) if draw_time else len(df)

    return {
        "game": game_cfg.id,
        "draw_time": draw_time,
        "total_draws": total,
        "window": window,
        "disclaimer": pred.DISCLAIMER,
        "heat_map": heat.to_dict(orient="records"),
        "suggestions": combos,
        "summary": summary,
    }


class AISummaryRequest(BaseModel):
    window: int | None = None


@router.post("/{game_id}/ai_summary")
def get_ai_summary(game_id: str, body: AISummaryRequest = AISummaryRequest()) -> dict:
    """Call Azure OpenAI to interpret the statistical patterns for a game."""
    from src.lotto.azure_ai import AzureAIClient

    game_cfg = GAMES.get(game_id.lower())
    if game_cfg is None:
        raise HTTPException(404, f"Game '{game_id}' not found")

    ai = AzureAIClient()
    if not ai.is_configured():
        raise HTTPException(503, ai.setup_instructions())

    df = _load_draws(game_cfg)
    if df.empty:
        raise HTTPException(404, f"No draw data for game '{game_id}'")

    _TIME_LABELS = {"14:00": "2:00 PM", "17:00": "5:00 PM", "21:00": "9:00 PM"}
    is_multi = len(game_cfg.draw_times) > 1

    if is_multi:
        slots_data = [
            {
                "label": _TIME_LABELS.get(dt, dt),
                "heat_map_rows": pred.number_heat_map(df, game_cfg, window=body.window, draw_time=dt).to_dict(orient="records"),
                "total_draws": int((df["draw_time"] == dt).sum()),
            }
            for dt in game_cfg.draw_times
        ]
    else:
        heat = pred.number_heat_map(df, game_cfg, window=body.window)
        slots_data = [
            {
                "label": _TIME_LABELS.get(game_cfg.draw_times[0], "9:00 PM"),
                "heat_map_rows": heat.to_dict(orient="records"),
                "total_draws": len(df),
            }
        ]

    chi_result = an.freq_chi_square(df, game_cfg).conclusion
    gap_result = an.gap_analysis(df, game_cfg).conclusion

    try:
        results = ai.interpret_lotto_stats(
            game_name=game_cfg.name,
            slots=slots_data,
            chi_sq_result=chi_result,
            gap_result=gap_result,
            picks=game_cfg.picks,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {
        "game": game_cfg.id,
        "results": results,
        "disclaimer": pred.DISCLAIMER,
    }

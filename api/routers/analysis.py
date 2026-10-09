from datetime import date
import sys
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
from api.db import get_connection
from src.lotto.games import GAMES
import src.lotto.analysis as an
import src.lotto.predict as pred
import src.lotto.lean_ml as lean_ml
import src.lotto.lucky as lucky

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


def _load_draws(game_cfg) -> "pd.DataFrame":
    import pandas as pd
    conn = get_connection(read_only=True)
    try:
        num_cols = ", ".join(f"n{i}" for i in range(1, game_cfg.picks + 1))
        df = conn.execute(f"""
            SELECT draw_date, weekday, game, draw_time,
                   pub_1, pub_2, pub_3, pub_4, pub_5, pub_6,
                   {num_cols}, jackpot_php, notes
            FROM draws WHERE game = ?
            ORDER BY draw_date, draw_time
        """, [game_cfg.id]).fetchdf()
    finally:
        conn.close()
    return df


def _generate(df, game_cfg, *, method, n, window, draw_time, hybrid=None) -> dict:
    """Use the same seeded local generation for displayed tickets and AI context."""
    if method == "hybrid":
        return lucky.hybrid_generation(
            df, game_cfg, n=n, window=window, draw_time=draw_time, seed=42, **(hybrid or {}),
        )
    if method == "ml":
        return lean_ml.generate_ml_combinations(
            df, game_cfg, n=n, window=window, draw_time=draw_time, seed=42,
        )
    return {
        "method": "uniform_fallback",
        "combinations": pred.suggest_combinations(
            df, game_cfg, n=n, window=window, draw_time=draw_time, seed=42,
        ),
        "scores": [], "feature_weights": {}, "training_draws": 0,
        "reason": "Uniform requested; no model was trained or evaluated.",
        "evaluation": {"status": "not_requested", "test_draws": 0,
                       "evidence_of_improvement": False},
    }


@router.get("/{game_id}/probability")
def get_probability(
    game_id: str,
    window: int | None = Query(default=None, gt=0, description="Limit to last N draws; omit for all"),
    n_suggestions: int = Query(default=5, ge=1, le=20),
    draw_time: str | None = Query(default=None, description="Filter to a specific draw time, e.g. '14:00'"),
    method: Literal["ml", "uniform", "hybrid"] = Query(default="ml", description=(
        "hybrid = ml_count numbers from the ML scores + the rest from lucky-number sources")),
    draw_date: date | None = Query(default=None, description="Draw date for lucky numbers; default next draw"),
    ml_count: int | None = Query(default=None, ge=0, description="Hybrid: numbers taken from ML (default picks // 2)"),
    birthdate: date | None = Query(default=None, description="Hybrid: optional birth date for personal numbers"),
    lucky_numbers: str | None = Query(default=None, description="Hybrid: your own lucky numbers, e.g. '7,23,41'"),
) -> dict:
    """Return historical heat maps and seeded local generation metadata.

    method=hybrid builds every ticket from ml_count ML numbers plus lucky numbers
    (Chinese day pillar / feng shui, numerology, birth date, your own numbers).

    For multi-slot ML requests without draw_time, by_draw_time maps each
    scheduled slot present in history to its generation (n_suggestions each).
    Top-level suggestions is empty and generation is null: no pooled model or
    implicit slot selection. Supply draw_time for one slot's suggestions.
    """
    game_cfg = GAMES.get(game_id.lower())
    if game_cfg is None:
        raise HTTPException(404, f"Game '{game_id}' not found")

    df = _load_draws(game_cfg)
    if df.empty:
        raise HTTPException(404, f"No draw data for game '{game_id}'")

    heat = pred.number_heat_map(df, game_cfg, window=window, draw_time=draw_time)
    by_draw_time = {}
    generation = None
    hybrid = {
        "draw_date": draw_date, "ml_count": ml_count, "birthdate": birthdate,
        "extra_numbers": lucky.parse_extra_numbers(lucky_numbers or "", game_cfg),
    } if method == "hybrid" else None
    if method in ("ml", "hybrid") and len(game_cfg.draw_times) > 1 and draw_time is None:
        for slot in game_cfg.draw_times:
            if df["draw_time"].eq(slot).any():
                by_draw_time[slot] = _generate(
                    df, game_cfg, method=method, n=n_suggestions, window=window, draw_time=slot,
                    hybrid=hybrid,
                )
    else:
        generation = _generate(
            df, game_cfg, method=method, n=n_suggestions, window=window, draw_time=draw_time,
            hybrid=hybrid,
        )
    summary = pred.heat_map_summary(heat)

    total = int((df["draw_time"] == draw_time).sum()) if draw_time else len(df)

    return {
        "game": game_cfg.id,
        "draw_time": draw_time,
        "total_draws": total,
        "window": window,
        "disclaimer": pred.DISCLAIMER,
        "heat_map": heat.to_dict(orient="records"),
        "suggestions": generation["combinations"] if generation is not None else [],
        "generation": generation,
        "by_draw_time": by_draw_time,
        "summary": summary,
    }


class AISummaryRequest(BaseModel):
    window: int | None = Field(default=None, gt=0)
    method: Literal["ml", "uniform", "hybrid"] = "ml"
    n_suggestions: int = Field(default=5, ge=1, le=20)
    # Hybrid only: lucky-number inputs (see /probability).
    draw_date: date | None = None
    ml_count: int | None = Field(default=None, ge=0)
    birthdate: date | None = None
    lucky_numbers: str | None = None


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
    slots_data = []
    for dt in game_cfg.draw_times:
        slot_df = df[df["draw_time"] == dt]
        if slot_df.empty:
            continue

        if game_cfg.ordered:
            heat_df = slot_df.tail(body.window) if body.window is not None else slot_df
            position_heat = pred.position_heat_map(heat_df, game_cfg)
            heat_map_rows = [
                {"number": int(row["digit"]), "position": int(position),
                 "total_draws": int(row["count"])}
                for position, heat in position_heat.items()
                for row in heat.to_dict(orient="records")
            ]
        else:
            heat = pred.number_heat_map(slot_df, game_cfg, window=body.window)
            heat_map_rows = heat.to_dict(orient="records")

        if not heat_map_rows:
            continue
        generation = _generate(
            df, game_cfg, method=body.method, n=body.n_suggestions,
            window=body.window, draw_time=dt,
            hybrid={"draw_date": body.draw_date, "ml_count": body.ml_count, "birthdate": body.birthdate,
                    "extra_numbers": lucky.parse_extra_numbers(body.lucky_numbers or "", game_cfg)}
            if body.method == "hybrid" else None,
        )
        # Add heat map summary for unordered games
        heat_summary_text = ""
        if not game_cfg.ordered and not heat.empty and "heat_score" in heat.columns:
            heat_summary_text = pred.heat_map_summary(heat)
        slots_data.append({
            "label": _TIME_LABELS.get(dt, dt),
            "heat_map_rows": heat_map_rows,
            "total_draws": min(len(slot_df), body.window) if game_cfg.ordered and body.window is not None else len(slot_df),
            "generation": generation,
            "heat_summary": heat_summary_text,
            "_heat_df": heat if not game_cfg.ordered else pd.DataFrame(),
        })

    if not slots_data:
        raise HTTPException(404, f"No draw data for game '{game_id}'")

    if game_cfg.ordered:
        chi_result = an.digit_uniformity(df, game_cfg).conclusion
        gap_result = an.digit_repeat_rate(df, game_cfg).conclusion
    else:
        chi_result = an.freq_chi_square(df, game_cfg).conclusion
        gap_result = an.gap_analysis(df, game_cfg).conclusion

    # Compute trend summary from the last computed heat map (unordered games only)
    _trend_summary = None
    if not game_cfg.ordered and slots_data:
        _last_heat_summary = slots_data[-1].get("_heat_df")
        if _last_heat_summary is not None and not _last_heat_summary.empty and "heat_score" in _last_heat_summary.columns:
            _trend_summary = {
                "heating_up": _last_heat_summary.nlargest(5, "heat_score")["number"].tolist(),
                "cooling_down": _last_heat_summary.nsmallest(5, "heat_score")["number"].tolist(),
            }
    # Remove internal _heat_df from slot dicts before passing to AI
    for _sd in slots_data:
        _sd.pop("_heat_df", None)

    try:
        results = ai.interpret_lotto_stats(
            game_name=game_cfg.name,
            slots=slots_data,
            chi_sq_result=chi_result,
            gap_result=gap_result,
            picks=game_cfg.picks,
            trend_summary=_trend_summary,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {
        "game": game_cfg.id,
        "results": results,
        "disclaimer": pred.DISCLAIMER,
    }


# Register the generic route after named routes so it cannot shadow probability.
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
    try:
        num_cols = ", ".join(f"n{i}" for i in range(1, game_cfg.picks + 1))
        df = conn.execute(f"""
            SELECT draw_date, weekday, game, draw_time,
                   pub_1, pub_2, pub_3, pub_4, pub_5, pub_6,
                   {num_cols}, jackpot_php, notes
            FROM draws WHERE game = ?
            ORDER BY draw_date
        """, [game_cfg.id]).fetchdf()
    finally:
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

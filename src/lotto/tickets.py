"""The standard way Hermes builds ticket combinations.

Every ticket is half ML, half lucky numbers:

* ML half: numbers drawn from the Lean ML scores (``lean_ml``), with ticket #1
  taking the highest-scoring numbers.
* Lucky half: numbers scored from the draw date (``lucky``): Chinese day pillar
  and feng shui, numerology, birth date, your own numbers and, optionally,
  numbers the AI suggested. Ticket #1 takes the highest-scoring ones the ML
  half did not take.

Later tickets sample both halves from their weights. The seed is fixed per draw
date and slot, so the same inputs give the same tickets; pass another seed to
get a fresh set. The dashboard and the API both call ``generate_tickets``; the
Strategy Arena measures this recipe against the alternatives (uniform, hot,
overdue, ML only, lucky only, crowd-avoiding).
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .games import GameConfig
from .lucky import hybrid_generation

STANDARD_METHOD = "hybrid_ml_lucky"


def generate_tickets(
    df, cfg: GameConfig, *, n: int = 5, window: int | None = None, draw_time: str | None = None,
    draw_date: dt.date | None = None, ml_count: int | None = None, birthdate: dt.date | None = None,
    extra_numbers: tuple[int, ...] | list[int] = (), ai_numbers: dict[int, str] | None = None,
    bet_feedback: dict[int, float] | None = None, seed: int = 42, crowd=None,
) -> dict:
    """Standard tickets for the next draw (see the module docstring).

    Returns hybrid_generation()'s dict: "tickets" (each with "numbers", "ml",
    "lucky"), "combinations", the ML evaluation and the lucky-number profile.
    With a fitted crowd model on a six-number game, each ticket also carries
    "crowd_index" (1.0 = average) and "share_kept" (share of a jackpot kept if
    it won, versus an average ticket).
    """
    result = hybrid_generation(
        df, cfg, n=n, window=window, draw_time=draw_time, draw_date=draw_date, ml_count=ml_count,
        birthdate=birthdate, extra_numbers=extra_numbers, ai_numbers=ai_numbers, seed=seed,
        bet_feedback=bet_feedback,
    )
    if crowd is not None and not cfg.ordered and cfg.id in crowd.game_ids and result["tickets"]:
        array = np.array(result["combinations"])
        for ticket, index, kept in zip(result["tickets"], crowd.crowd_index(array, cfg.id),
                                       crowd.share_factor(array, cfg.id)):
            ticket["crowd_index"], ticket["share_kept"] = float(index), float(kept)
    return result

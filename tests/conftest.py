"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
import pandas as pd
from src.lotto.games import GAMES, GameConfig


@pytest.fixture
def game_655() -> GameConfig:
    return GAMES["655"]


@pytest.fixture
def game_3d() -> GameConfig:
    return GAMES["3d"]


def _fair_draws(game_cfg: GameConfig, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Generate n fair simulated draws for a pick-6 game."""
    rng = np.random.default_rng(seed)
    pool = np.arange(game_cfg.low_boundary, game_cfg.pool + 1)
    rows = []
    from datetime import date, timedelta
    start = date(2020, 1, 1)
    draw_days = list(game_cfg.draw_days)
    day_abbr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d = start
    count = 0
    while count < n:
        abbr = d.strftime("%a")
        if abbr in draw_days:
            nums = sorted(rng.choice(pool, size=game_cfg.picks, replace=not game_cfg.ordered).tolist())
            row = {"draw_date": d.isoformat(), "weekday": abbr, "game": game_cfg.id,
                   "draw_time": "21:00", "jackpot_php": 50_000_000.0, "notes": None}
            for i, v in enumerate(nums, 1):
                row[f"pub_{i}"] = v
                row[f"n{i}"] = v
            rows.append(row)
            count += 1
        d += timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture
def fair_draws_655(game_655: GameConfig) -> pd.DataFrame:
    return _fair_draws(game_655, n=500, seed=42)

"""Single source of truth for all PCSO game definitions."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GameConfig:
    id: str                    # short slug used as DB key
    name: str                  # display name
    pool: int                  # highest number in pool
    picks: int                 # how many numbers drawn
    draw_days: tuple[str, ...]  # Mon Tue Wed Thu Fri Sat Sun
    draw_times: tuple[str, ...]  # 24h strings, e.g. "21:00"
    ordered: bool              # True = digits in exact order (6D, 4D, 3D, 2D)
    low_boundary: int = 1      # lowest number; digits games start at 0


GAMES: dict[str, GameConfig] = {
    "658": GameConfig(
        id="658",
        name="6/58 Ultra Lotto",
        pool=58,
        picks=6,
        draw_days=("Tue", "Fri", "Sun"),
        draw_times=("21:00",),
        ordered=False,
    ),
    "655": GameConfig(
        id="655",
        name="6/55 Grand Lotto",
        pool=55,
        picks=6,
        draw_days=("Mon", "Wed", "Sat"),
        draw_times=("21:00",),
        ordered=False,
    ),
    "649": GameConfig(
        id="649",
        name="6/49 Super Lotto",
        pool=49,
        picks=6,
        draw_days=("Tue", "Thu", "Sun"),
        draw_times=("21:00",),
        ordered=False,
    ),
    "645": GameConfig(
        id="645",
        name="6/45 Mega Lotto",
        pool=45,
        picks=6,
        draw_days=("Mon", "Wed", "Fri"),
        draw_times=("21:00",),
        ordered=False,
    ),
    "642": GameConfig(
        id="642",
        name="6/42 Lotto",
        pool=42,
        picks=6,
        draw_days=("Tue", "Thu", "Sat"),
        draw_times=("21:00",),
        ordered=False,
    ),
    "6d": GameConfig(
        id="6d",
        name="6D Lotto",
        pool=9,
        picks=6,
        draw_days=("Tue", "Thu", "Sat"),
        draw_times=("21:00",),
        ordered=True,
        low_boundary=0,
    ),
    "4d": GameConfig(
        id="4d",
        name="4D Lotto",
        pool=9,
        picks=4,
        draw_days=("Mon", "Wed", "Fri"),
        draw_times=("21:00",),
        ordered=True,
        low_boundary=0,
    ),
    "3d": GameConfig(
        id="3d",
        name="3D Swertres",
        pool=9,
        picks=3,
        draw_days=("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
        draw_times=("14:00", "17:00", "21:00"),
        ordered=True,
        low_boundary=0,
    ),
    "2d": GameConfig(
        id="2d",
        name="2D EZ2",
        pool=31,
        picks=2,
        draw_days=("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
        draw_times=("14:00", "17:00", "21:00"),
        ordered=True,
        low_boundary=1,
    ),
}


def get_game(game_id: str) -> GameConfig:
    """Return a GameConfig by id; raises KeyError on unknown id."""
    return GAMES[game_id.lower()]


def pick6_games() -> list[GameConfig]:
    return [g for g in GAMES.values() if not g.ordered and g.picks == 6]


def digit_games() -> list[GameConfig]:
    return [g for g in GAMES.values() if g.ordered]

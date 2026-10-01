"""
Import PCSO lotto data from Kaggle dataset:
  harrychristian/pcso-lottery-results-philippines

Handles the dataset's column naming and combination string formats,
maps to our game IDs, and loads via the standard clean/insert pipeline.

Usage (CLI):
    python -m src.lotto kaggle_import

Usage (Python):
    from src.lotto.kaggle_import import import_kaggle_dataset
    inserted, skipped = import_kaggle_dataset()
"""
from __future__ import annotations

import re
import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Game name → our game ID
# ---------------------------------------------------------------------------

_GAME_NAME_MAP: dict[str, str] = {
    # 6/58
    "6/58": "658", "658": "658", "ultra lotto": "658", "ultra": "658",
    "6-58": "658", "6_58": "658",
    # 6/55
    "6/55": "655", "655": "655", "grand lotto": "655",
    "6-55": "655", "6_55": "655",
    # 6/49
    "6/49": "649", "649": "649", "super lotto": "649", "superlotto": "649",
    "6-49": "649", "6_49": "649",
    # 6/45
    "6/45": "645", "645": "645", "mega lotto": "645", "megalotto": "645",
    "6-45": "645", "6_45": "645",
    # 6/42
    "6/42": "642", "642": "642", "lotto 6/42": "642",
    "6-42": "642", "6_42": "642",
    # digit games (limited support — no sorted n-cols needed)
    "4d": "4d", "4-d": "4d", "4d lotto": "4d",
    "3d": "3d", "3-d": "3d", "swertres": "3d", "3d lotto": "3d",
    "2d": "2d", "2-d": "2d", "ez2": "2d", "2d lotto": "2d",
    "6d": "6d", "6-d": "6d",
}

_DRAW_TIME_MAP: dict[str, str] = {
    "2pm": "14:00",
    "5pm": "17:00",
    "9pm": "21:00",
    "11am": "11:00",
}


def _extract_draw_time(raw_game: str) -> str:
    """Extract draw time from game name suffix (e.g. '3D Lotto 9PM' → '21:00')."""
    lower = raw_game.strip().lower()
    for suffix, t in _DRAW_TIME_MAP.items():
        if lower.endswith(suffix):
            return t
    return "21:00"


def _normalize_game(raw: str) -> str | None:
    """Map raw game name string to our game ID. Returns None if unknown."""
    key = str(raw).strip().lower()
    # exact match
    if key in _GAME_NAME_MAP:
        return _GAME_NAME_MAP[key]
    # partial match — e.g. "Lotto 6/42 result" → "642"
    for pattern, gid in _GAME_NAME_MAP.items():
        if pattern in key:
            return gid
    return None


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

def _parse_numbers(raw: Any) -> list[int] | None:
    """
    Parse a combination string or list into a sorted list of ints.
    Handles: "5-12-23-34-41-53", "5, 12, 23", "05 12 23 34 41 53", [5, 12, ...]
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (list, tuple)):
        try:
            return sorted(int(n) for n in raw)
        except (ValueError, TypeError):
            return None
    nums = re.findall(r"\d+", str(raw))
    if not nums:
        return None
    try:
        return sorted(int(n) for n in nums)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Column name normalisation
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name (case-insensitive)."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------

def import_kaggle_dataset(
    kaggle_handle: str = "harrychristian/pcso-lottery-results-philippines",
    file_path: str = "",
    conn=None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Load the Kaggle dataset and insert rows into the local DuckDB.

    Returns (total_inserted, total_skipped).
    Skipped = rows that were duplicates, had invalid data, or unknown game.
    """
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    from .games import GAMES
    from .clean import load_and_clean

    close_conn = False
    if conn is None and not dry_run:
        from api.db import get_connection
        conn = get_connection()
        close_conn = True

    try:
        logger.info("Loading Kaggle dataset %s …", kaggle_handle)
        raw_df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            kaggle_handle,
            file_path,
        )
        logger.info("Loaded %d rows, columns: %s", len(raw_df), list(raw_df.columns))

        # Normalise column names to lowercase for detection
        raw_df.columns = [c.strip() for c in raw_df.columns]

        rows = _convert_dataframe(raw_df)
        logger.info("Converted %d parseable rows", len(rows))

        if dry_run:
            return len(rows), 0

        total_inserted = total_skipped = 0
        by_game: dict[str, list[dict]] = {}
        for row in rows:
            gid = row.get("game")
            if gid not in GAMES:
                total_skipped += 1
                continue
            by_game.setdefault(gid, []).append(row)

        for gid, game_rows in by_game.items():
            ins, flg = load_and_clean(game_rows, GAMES[gid], conn)
            total_inserted += ins
            total_skipped += flg
            logger.info("game=%s inserted=%d skipped=%d", gid, ins, flg)

        return total_inserted, total_skipped

    finally:
        if close_conn and conn is not None:
            conn.close()


def _convert_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a raw Kaggle DataFrame to a list of row dicts compatible
    with validate_row / load_and_clean.

    Handles multiple possible column naming conventions.
    """
    rows: list[dict[str, Any]] = []

    # --- Detect key columns ---
    date_col = _find_col(df, [
        "draw date", "draw_date", "date", "drawdate", "lotto draw date",
    ])
    game_col = _find_col(df, [
        "lotto game", "game", "game type", "type", "lotto type",
    ])
    combo_col = _find_col(df, [
        "winning combination", "combination", "combinations", "winning numbers",
        "numbers", "result", "winning_combination",
    ])
    jackpot_col = _find_col(df, [
        "jackpot", "jackpot prize", "prize", "jackpot_php",
        "jackpot (php)", "prize (php)",
    ])
    winners_col = _find_col(df, [
        "winners", "no. of winners", "number of winners", "jackpot winners",
    ])

    if date_col is None:
        raise ValueError(
            f"Cannot find a date column. Available columns: {list(df.columns)}"
        )
    if combo_col is None and game_col is None:
        raise ValueError(
            f"Cannot find combination or game columns. Available: {list(df.columns)}"
        )

    for _, row in df.iterrows():
        # --- Parse date ---
        raw_date = row.get(date_col) if date_col else None
        draw_date = _parse_date(raw_date)
        if draw_date is None:
            continue

        # --- Parse game ---
        raw_game = row.get(game_col) if game_col else _infer_game_from_filename(df)
        game_id = _normalize_game(str(raw_game)) if raw_game else None
        if game_id is None:
            continue

        # --- Parse numbers ---
        raw_combo = row.get(combo_col) if combo_col else None
        nums = _parse_numbers(raw_combo)
        if nums is None:
            continue

        from .games import GAMES
        gcfg = GAMES.get(game_id)
        if gcfg is None:
            continue

        picks = gcfg.picks
        if len(nums) < picks:
            continue  # not enough numbers
        nums = nums[:picks]  # trim extras if any

        # Build pub_* (draw order, same as sorted for pick-6)
        pub = {f"pub_{i+1}": nums[i] for i in range(picks)}
        # n* (sorted ascending) for unordered games
        if not gcfg.ordered:
            n_sorted = sorted(nums)
            n_cols = {f"n{i+1}": n_sorted[i] for i in range(picks)}
        else:
            n_cols = {f"n{i+1}": nums[i] for i in range(picks)}

        # --- Parse jackpot ---
        jackpot = None
        if jackpot_col:
            try:
                j_raw = str(row[jackpot_col]).replace(",", "").replace("₱", "").strip()
                jackpot = float(j_raw)
                if jackpot <= 0:
                    jackpot = None
            except (ValueError, TypeError):
                jackpot = None

        # --- Parse winners ---
        winners_jackpot = None
        if winners_col:
            try:
                w_raw = str(row[winners_col]).strip()
                winners_jackpot = int(float(w_raw)) if w_raw not in ("", "nan", "None") else None
            except (ValueError, TypeError):
                winners_jackpot = None

        draw_time = _extract_draw_time(str(raw_game)) if raw_game else "21:00"
        weekday = draw_date.strftime("%a")
        draw_row: dict[str, Any] = {
            "draw_date": draw_date.isoformat(),
            "weekday": weekday,
            "game": game_id,
            "draw_time": draw_time,
            "jackpot_php": jackpot,
            "winners_jackpot": winners_jackpot,
            "notes": None,
        }
        draw_row.update(pub)
        draw_row.update(n_cols)
        rows.append(draw_row)

    return rows


def _parse_date(raw: Any) -> date | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, (date, datetime)):
        return raw.date() if isinstance(raw, datetime) else raw
    if isinstance(raw, pd.Timestamp):
        return raw.date()

    raw_str = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
                "%b %d, %Y", "%d-%b-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw_str, fmt).date()
        except ValueError:
            continue
    return None


def _infer_game_from_filename(df: pd.DataFrame) -> str | None:
    """Last-resort: try to read game from DataFrame attrs (set by file name)."""
    src = getattr(df, "attrs", {}).get("source", "")
    return _normalize_game(src) if src else None


# ---------------------------------------------------------------------------
# Local CSV import
# ---------------------------------------------------------------------------

def import_csv_file(
    file_path: str,
    conn=None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Import a local CSV file with PCSO draw data into DuckDB.

    Expected columns (case-insensitive):
        LOTTO GAME, COMBINATIONS, DRAW DATE, JACKPOT (PHP), WINNERS

    Returns (total_inserted, total_skipped).
    """
    from .games import GAMES
    from .clean import load_and_clean

    close_conn = False
    if conn is None and not dry_run:
        from api.db import get_connection
        conn = get_connection()
        close_conn = True

    try:
        df = pd.read_csv(file_path, index_col=0)
        df.columns = [c.strip() for c in df.columns]
        logger.info("Loaded %d rows from %s, columns: %s", len(df), file_path, list(df.columns))

        rows = _convert_dataframe(df)
        logger.info("Converted %d parseable rows", len(rows))

        if dry_run:
            return len(rows), 0

        total_inserted = total_skipped = 0
        by_game: dict[str, list[dict]] = {}
        for row in rows:
            gid = row.get("game")
            if gid not in GAMES:
                total_skipped += 1
                continue
            by_game.setdefault(gid, []).append(row)

        for gid, game_rows in by_game.items():
            ins, flg = load_and_clean(game_rows, GAMES[gid], conn)
            total_inserted += ins
            total_skipped += flg
            logger.info("game=%s inserted=%d skipped=%d", gid, ins, flg)

        return total_inserted, total_skipped

    finally:
        if close_conn and conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Inspect helper — call this first to see what columns the dataset has
# ---------------------------------------------------------------------------

def inspect_kaggle_dataset(
    kaggle_handle: str = "harrychristian/pcso-lottery-results-philippines",
    file_path: str = "",
) -> pd.DataFrame:
    """
    Load the dataset and print its columns + first 5 rows.
    Run this before import to verify the column mapping.
    """
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        kaggle_handle,
        file_path,
    )
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    return df

"""
Database migration and enrichment utilities.

Adds derived columns to the draws table and backfills them from existing data.
Safe to run multiple times — uses ADD COLUMN IF NOT EXISTS pattern (via try/except).

New columns added:
  draw_sum        SMALLINT  — sum of all drawn numbers
  odd_count       TINYINT   — how many odd numbers in the draw
  high_count      TINYINT   — how many numbers above pool midpoint
  consecutive_cnt TINYINT   — count of consecutive number pairs (e.g. 5-6 = 1 pair)
  number_spread   SMALLINT  — max_number - min_number
  is_rollover     BOOLEAN   — True if jackpot grew from previous draw (no winner last time)
  covid_gap       BOOLEAN   — True if draw immediately follows the 2020 COVID suspension
  winners_jackpot SMALLINT  — number of jackpot winners (NULL until scraped)
  winners_5match  SMALLINT  — number of 5-match winners (NULL until scraped)
  winners_4match  SMALLINT  — number of 4-match winners (NULL until scraped)
  winners_3match  SMALLINT  — number of 3-match winners (NULL until scraped)
"""
from __future__ import annotations

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Game start dates (PCSO official launch dates)
# ---------------------------------------------------------------------------
GAME_START_YEARS: dict[str, int] = {
    "2d":  1995,
    "3d":  1995,
    "642": 1995,
    "645": 1996,
    "649": 1998,
    "4d":  2005,
    "655": 2004,
    "658": 2010,
    "6d":  2010,
}


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

_NEW_COLUMNS = [
    ("draw_sum",        "SMALLINT"),
    ("odd_count",       "TINYINT"),
    ("high_count",      "TINYINT"),
    ("consecutive_cnt", "TINYINT"),
    ("number_spread",   "SMALLINT"),
    ("is_rollover",     "BOOLEAN"),
    ("covid_gap",       "BOOLEAN"),
    ("winners_jackpot", "SMALLINT"),
    ("winners_5match",  "SMALLINT"),
    ("winners_4match",  "SMALLINT"),
    ("winners_3match",  "SMALLINT"),
]


def add_derived_columns(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """
    Add new columns to the draws table if they don't exist.
    Returns list of columns actually added.
    """
    # Get existing columns
    existing = {
        row[0].lower()
        for row in conn.execute("DESCRIBE draws").fetchall()
    }

    added = []
    for col_name, col_type in _NEW_COLUMNS:
        if col_name.lower() not in existing:
            conn.execute(f"ALTER TABLE draws ADD COLUMN {col_name} {col_type} DEFAULT NULL")
            added.append(col_name)

    return added


# ---------------------------------------------------------------------------
# Backfill derived columns from existing number data
# ---------------------------------------------------------------------------

def backfill_derived_columns(
    conn: duckdb.DuckDBPyConnection,
    game: str | None = None,
) -> dict[str, int]:
    """
    Compute and populate draw_sum, odd_count, high_count, consecutive_cnt,
    number_spread, is_rollover, covid_gap from existing data.

    Pass game='655' to limit to one game, or None for all games.
    Returns dict with counts of rows updated per column.
    """
    from .games import GAMES

    where = f"WHERE game = '{game}'" if game else ""
    df = conn.execute(f"""
        SELECT draw_date, game, draw_time,
               n1, n2, n3, n4, n5, n6,
               pub_1, pub_2, pub_3,
               jackpot_php
        FROM draws {where}
        ORDER BY game, draw_date, draw_time
    """).fetchdf()

    if df.empty:
        return {}

    results = []

    for (gid, dt_val), grp in df.groupby(["game", "draw_time"], sort=True):
        game_cfg = GAMES.get(gid)
        if game_cfg is None:
            continue

        picks = game_cfg.picks
        pool_mid = (game_cfg.low_boundary + game_cfg.pool) / 2.0

        # Number columns (sorted = n-cols for unordered, pub-cols for ordered)
        if game_cfg.ordered:
            num_cols = [f"pub_{i}" for i in range(1, picks + 1)]
        else:
            num_cols = [f"n{i}" for i in range(1, picks + 1)]

        available_cols = [c for c in num_cols if c in grp.columns]
        if not available_cols:
            continue

        # Jackpot lag for is_rollover (per draw_time stream)
        grp = grp.sort_values("draw_date").copy()
        grp["jackpot_prev"] = grp["jackpot_php"].shift(1)

        for _, row in grp.iterrows():
            nums = [row[c] for c in available_cols if pd.notna(row[c])]
            nums_int = sorted(int(n) for n in nums)

            if not nums_int:
                continue

            draw_sum = sum(nums_int)
            odd_count = sum(1 for n in nums_int if n % 2 == 1)
            high_count = sum(1 for n in nums_int if n > pool_mid)
            consec = sum(
                1 for i in range(len(nums_int) - 1)
                if nums_int[i + 1] - nums_int[i] == 1
            )
            spread = nums_int[-1] - nums_int[0] if len(nums_int) > 1 else 0

            # is_rollover: jackpot grew vs previous draw in same game+draw_time
            jackpot = row.get("jackpot_php")
            prev_jp = row.get("jackpot_prev")
            if pd.notna(jackpot) and pd.notna(prev_jp) and jackpot > prev_jp:
                is_rollover = True
            elif pd.notna(jackpot) and pd.notna(prev_jp):
                is_rollover = False
            else:
                is_rollover = None

            # covid_gap: first draw after the 2020 suspension
            draw_date = pd.to_datetime(row["draw_date"]).date()
            covid_gap = (
                pd.to_datetime("2020-03-15").date() <= draw_date
                <= pd.to_datetime("2020-10-31").date()
            )

            results.append({
                "draw_date": str(row["draw_date"])[:10],
                "game": gid,
                "draw_time": dt_val,
                "draw_sum": draw_sum,
                "odd_count": odd_count,
                "high_count": high_count,
                "consecutive_cnt": consec,
                "number_spread": spread,
                "is_rollover": is_rollover,
                "covid_gap": bool(covid_gap),
            })

    if not results:
        return {}

    updates_df = pd.DataFrame(results)

    # Register as temp table and bulk UPDATE
    conn.register("_backfill_tmp", updates_df)
    conn.execute("""
        UPDATE draws
        SET
            draw_sum        = _backfill_tmp.draw_sum,
            odd_count       = _backfill_tmp.odd_count,
            high_count      = _backfill_tmp.high_count,
            consecutive_cnt = _backfill_tmp.consecutive_cnt,
            number_spread   = _backfill_tmp.number_spread,
            is_rollover     = _backfill_tmp.is_rollover,
            covid_gap       = _backfill_tmp.covid_gap
        FROM _backfill_tmp
        WHERE draws.draw_date  = CAST(_backfill_tmp.draw_date AS DATE)
          AND draws.game       = _backfill_tmp.game
          AND (draws.draw_time = _backfill_tmp.draw_time
               OR (draws.draw_time IS NULL AND _backfill_tmp.draw_time IS NULL))
    """)
    conn.unregister("_backfill_tmp")

    n_updated = len(results)
    return {"rows_updated": n_updated}


# ---------------------------------------------------------------------------
# Historical data scraping (pre-2016)
# ---------------------------------------------------------------------------

async def scrape_historical(
    game_id: str,
    from_year: int | None = None,
    to_year: int | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
    progress_cb=None,
) -> dict:
    """
    Scrape historical draw data from lottopcso.com for years before 2016.

    Parameters
    ----------
    game_id    : game id, e.g. '655'
    from_year  : start year (default: GAME_START_YEARS[game_id])
    to_year    : end year inclusive (default: 2015)
    conn       : writable DuckDB connection (if None, opens one)
    progress_cb: optional callable(year, inserted) for progress reporting

    Returns dict with total_inserted, total_flagged, years_scraped.
    """
    import asyncio
    from datetime import datetime
    from .games import GAMES
    from .scrape import scrape_pick6
    from .clean import load_and_clean

    game_cfg = GAMES.get(game_id)
    if game_cfg is None:
        raise ValueError(f"Unknown game: {game_id}")

    if game_cfg.ordered:
        raise ValueError("Historical scraping not supported for ordered games (2D/3D/4D/6D) via lottopcso.com — use pcso.gov.ph scraper instead.")

    start_year = from_year or GAME_START_YEARS.get(game_id, 2010)
    end_year = to_year or 2015

    years = list(range(start_year, end_year + 1))

    close_conn = False
    if conn is None:
        from api.db import get_connection
        conn = get_connection()
        close_conn = True

    total_inserted = 0
    total_flagged = 0
    years_scraped = []

    try:
        for year in years:
            raw = await scrape_pick6(game_cfg, [year])
            if raw:
                ins, flg = load_and_clean(raw, game_cfg, conn)
                total_inserted += ins
                total_flagged += flg
                years_scraped.append(year)
                if progress_cb:
                    progress_cb(year, ins)
            else:
                if progress_cb:
                    progress_cb(year, 0)
    finally:
        if close_conn:
            conn.close()

    return {
        "total_inserted": total_inserted,
        "total_flagged": total_flagged,
        "years_scraped": years_scraped,
    }


def run_full_migration(conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Run the complete migration: add columns + backfill.
    Safe to run multiple times.
    """
    added = add_derived_columns(conn)
    updated = backfill_derived_columns(conn)
    return {"columns_added": added, "rows_updated": updated.get("rows_updated", 0)}

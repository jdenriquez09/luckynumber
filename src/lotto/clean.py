"""Data validation, flagging, and loading into DuckDB.

Rules (from CLAUDE.md):
- Date must fall on a valid draw weekday for the game — flag + note if not.
- Numbers must be `picks` unique integers within [low_boundary, pool] — blank row + note if not.
- Jackpot must be a plausible positive float — blank (not guess) if malformed.
- Duplicate (draw_date, game, draw_time) keys are errors — skip with note.
- Never silently fix: only correct when evidence exists; record fix in notes.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from .games import GameConfig

_WEEKDAY_MAP = {
    "Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6,
}
_ISOWEEKDAY_TO_ABBR = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def _valid_draw_weekday(d: date, game_cfg: GameConfig) -> bool:
    abbr = _ISOWEEKDAY_TO_ABBR[d.isoweekday()]
    return abbr in game_cfg.draw_days


def _validate_jackpot(raw: Any) -> tuple[float | None, str | None]:
    if raw is None:
        return None, None
    try:
        v = float(raw)
        if v <= 0:
            return None, f"jackpot non-positive ({raw!r})"
        return v, None
    except (TypeError, ValueError):
        return None, f"jackpot unparseable ({raw!r})"


def validate_row(row: dict[str, Any], game_cfg: GameConfig) -> dict[str, Any]:
    """
    Validate a raw parsed row dict.
    Returns a cleaned copy; invalid fields are set to None.
    notes field accumulates pipe-separated problem descriptions.
    """
    out = dict(row)
    notes: list[str] = []

    # --- draw_date ---
    try:
        d = date.fromisoformat(str(row["draw_date"]))
    except (ValueError, TypeError):
        notes.append(f"unparseable draw_date {row.get('draw_date')!r}")
        out["draw_date"] = None
        d = None

    if d is not None and not _valid_draw_weekday(d, game_cfg):
        notes.append(
            f"draw_date {d} is {_ISOWEEKDAY_TO_ABBR[d.isoweekday()]} "
            f"but {game_cfg.id} draws on {','.join(game_cfg.draw_days)}"
        )
        # do NOT blank draw_date — flag only; the slug may confirm the true date

    # --- numbers ---
    pub_keys = [f"pub_{i}" for i in range(1, game_cfg.picks + 1)]
    nums = [row.get(k) for k in pub_keys]
    nums_valid = all(n is not None for n in nums)

    if nums_valid:
        try:
            int_nums = [int(n) for n in nums]  # type: ignore[arg-type]
        except (ValueError, TypeError):
            nums_valid = False
            notes.append(f"non-integer in numbers: {nums!r}")
            int_nums = []

        if nums_valid:
            # range check
            bad_range = [n for n in int_nums if not (game_cfg.low_boundary <= n <= game_cfg.pool)]
            if bad_range:
                nums_valid = False
                notes.append(
                    f"numbers out of range [{game_cfg.low_boundary},{game_cfg.pool}]: {bad_range}"
                )

            # uniqueness (pick-6 only; digit games allow repeats)
            if nums_valid and not game_cfg.ordered:
                if len(set(int_nums)) < len(int_nums):
                    nums_valid = False
                    notes.append(f"duplicate numbers: {int_nums}")

            if nums_valid and not game_cfg.ordered:
                sorted_nums = sorted(int_nums)
                for i, k in enumerate(["n1", "n2", "n3", "n4", "n5", "n6"][: game_cfg.picks]):
                    out[k] = sorted_nums[i]
            elif nums_valid and game_cfg.ordered:
                for i, k in enumerate(["n1", "n2", "n3", "n4", "n5", "n6"][: game_cfg.picks]):
                    out[k] = int_nums[i]
    else:
        notes.append("missing numbers")

    if not nums_valid:
        for k in pub_keys + ["n1", "n2", "n3", "n4", "n5", "n6"]:
            out[k] = None

    # --- jackpot ---
    jackpot_val, jackpot_note = _validate_jackpot(row.get("jackpot_php"))
    out["jackpot_php"] = jackpot_val
    if jackpot_note:
        notes.append(jackpot_note)

    # --- notes ---
    existing = out.get("notes") or ""
    combined = " | ".join(filter(None, [existing] + notes))
    out["notes"] = combined if combined else None

    return out


def load_and_clean(
    raw_rows: list[dict[str, Any]],
    game_cfg: GameConfig,
    conn: Any,
) -> tuple[int, int]:
    """
    Validate raw_rows and upsert valid ones into DuckDB.
    Returns (inserted_count, flagged_count).
    """
    inserted = 0
    flagged = 0

    for row in raw_rows:
        cleaned = validate_row(row, game_cfg)

        if cleaned.get("draw_date") is None:
            flagged += 1
            continue

        draw_time = cleaned.get("draw_time") or "21:00"

        try:
            conn.execute("""
                INSERT INTO draws (
                    draw_date, weekday, game, draw_time,
                    pub_1, pub_2, pub_3, pub_4, pub_5, pub_6,
                    n1, n2, n3, n4, n5, n6,
                    jackpot_php, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (draw_date, game, draw_time) DO NOTHING
            """, [
                cleaned["draw_date"],
                cleaned.get("weekday"),
                game_cfg.id,
                draw_time,
                cleaned.get("pub_1"), cleaned.get("pub_2"),
                cleaned.get("pub_3"), cleaned.get("pub_4"),
                cleaned.get("pub_5"), cleaned.get("pub_6"),
                cleaned.get("n1"), cleaned.get("n2"),
                cleaned.get("n3"), cleaned.get("n4"),
                cleaned.get("n5"), cleaned.get("n6"),
                cleaned.get("jackpot_php"),
                cleaned.get("notes"),
            ])
            inserted += 1
        except Exception as exc:
            flagged += 1
            import logging
            logging.getLogger(__name__).warning("insert failed: %s | row=%s", exc, cleaned)

    return inserted, flagged


def completeness_report(game_cfg: GameConfig, conn: Any) -> dict[str, Any]:
    """
    Return coverage stats and list unexplained calendar gaps.
    """
    import pandas as pd
    from datetime import timedelta

    rows = conn.execute("""
        SELECT draw_date FROM draws
        WHERE game = ?
        ORDER BY draw_date
    """, [game_cfg.id]).fetchdf()

    if rows.empty:
        return {"game": game_cfg.id, "count": 0, "first": None, "last": None, "gaps": []}

    dates = pd.to_datetime(rows["draw_date"])
    first, last = dates.min().date(), dates.max().date()
    actual = set(d.date() for d in dates)

    # expected draw dates between first and last
    expected: list[date] = []
    cur = first
    while cur <= last:
        if _ISOWEEKDAY_TO_ABBR[cur.isoweekday()] in game_cfg.draw_days:
            expected.append(cur)
        cur += timedelta(days=1)

    gaps = sorted(set(expected) - actual)

    return {
        "game": game_cfg.id,
        "count": len(actual),
        "first": first.isoformat(),
        "last": last.isoformat(),
        "expected": len(expected),
        "gaps": [g.isoformat() for g in gaps],
    }

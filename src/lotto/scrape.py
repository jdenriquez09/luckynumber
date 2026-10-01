"""Async scraper for lottopcso.com and pcso.gov.ph with caching and rate-limiting."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .games import GameConfig, GAMES

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RATE_LIMIT_S = 1.1  # seconds between requests
USER_AGENT = (
    "Mozilla/5.0 (compatible; PCSOLottoScraper/1.0; "
    "+https://github.com/user/luckynumber)"
)

_last_request_time: float = 0.0


# ---------------------------------------------------------------------------
# Low-level HTTP + cache
# ---------------------------------------------------------------------------

def _cache_path(url: str) -> Path:
    slug = hashlib.sha1(url.encode()).hexdigest()[:16]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{slug}.html"


def _load_cache(url: str) -> str | None:
    p = _cache_path(url)
    if p.exists():
        logger.debug("cache hit: %s", url)
        return p.read_text(encoding="utf-8")
    return None


def _save_cache(url: str, html: str) -> Path:
    p = _cache_path(url)
    p.write_text(html, encoding="utf-8")
    return p


async def fetch_page(url: str, *, force: bool = False) -> tuple[str, Path | None]:
    """Return (html, cache_path). Uses cache unless force=True."""
    global _last_request_time

    if not force:
        cached = _load_cache(url)
        if cached is not None:
            return cached, _cache_path(url)

    # rate-limit
    elapsed = time.monotonic() - _last_request_time
    if elapsed < RATE_LIMIT_S:
        await asyncio.sleep(RATE_LIMIT_S - elapsed)

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=30,
    ) as client:
        resp = await client.get(url)
        _last_request_time = time.monotonic()
        logger.info("GET %s → %s", url, resp.status_code)
        resp.raise_for_status()
        html = resp.text

    p = _save_cache(url, html)
    return html, p


# ---------------------------------------------------------------------------
# Parser: lottopcso.com pick-6 history tables
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date(raw: str) -> date | None:
    raw = raw.strip()
    # Handles: "Sep. 20, 2026" | "July 31, 2026" | "Mar 31, 2026" | "April 28 2026"
    m = re.search(r"(\w+)\.?\s+(\d{1,2}),?\s+(\d{4})", raw, re.I)
    if m:
        month = _MONTH_MAP.get(m.group(1)[:3].lower())
        if month:
            try:
                return date(int(m.group(3)), month, int(m.group(2)))
            except ValueError:
                pass
    # ISO fallback: 2025-01-06
    m2 = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m2:
        try:
            return date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        except ValueError:
            pass
    # MM/DD/YYYY (pcso.gov.ph format: "3/3/2026")
    m3 = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m3:
        try:
            return date(int(m3.group(3)), int(m3.group(1)), int(m3.group(2)))
        except ValueError:
            pass
    return None


def _parse_jackpot(raw: str) -> float | None:
    cleaned = re.sub(r"[^\d.]", "", raw.replace(",", ""))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _parse_winner_table(soup, game_cfg: "GameConfig") -> dict[str, int]:
    """
    Parse the 6-column jackpot-winners table on lottopcso.com pages.
    Returns {date_iso: winner_count} filtered to the given game.
    Table headers: Date | Draw | Combination | Jackpot | Location | Winner
    The "Draw" column contains a game label like "6/49" or "6/42".
    """
    # Build a set of game label fragments that match game_cfg
    # e.g. game_cfg.id "649" → look for "6/49" in the Draw cell
    picks_str = game_cfg.id  # "649", "655", "658", ...
    # Convert "649" → "6/49", "655" → "6/55", etc.
    game_label_variants: set[str] = set()
    if picks_str.isdigit() and len(picks_str) == 3:
        game_label_variants.add(picks_str[0] + "/" + picks_str[1:])  # "6/49"
        game_label_variants.add(picks_str)
    else:
        game_label_variants.add(picks_str.lower())

    winners_by_date: dict[str, int] = {}
    for tbl in soup.find_all("table"):
        header_row = tbl.find("tr")
        if not header_row:
            continue
        headers = [th.get_text(" ", strip=True).lower()
                   for th in header_row.find_all(["th", "td"])]
        # Must have 'date', 'combination', 'draw', and 'winner' columns
        if not (any("date" in h for h in headers)
                and any("combination" in h for h in headers)
                and any("winner" in h for h in headers)):
            continue
        # Find column indices
        date_idx = next((i for i, h in enumerate(headers) if "date" in h), 0)
        draw_idx = next((i for i, h in enumerate(headers) if h == "draw"), None)
        winner_idx = next((i for i, h in enumerate(headers) if "winner" in h), -1)
        if winner_idx < 0:
            continue
        for tr in tbl.find_all("tr")[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) <= winner_idx:
                continue
            # Filter to matching game if "draw" column exists
            if draw_idx is not None and draw_idx < len(cells):
                cell_game = cells[draw_idx].strip()
                if not any(v in cell_game for v in game_label_variants):
                    continue
            d = _parse_date(cells[date_idx])
            if d is None:
                continue
            try:
                count = int(re.sub(r"[^\d]", "", cells[winner_idx]) or "0")
            except (ValueError, TypeError):
                count = 0
            iso = d.isoformat()
            winners_by_date[iso] = winners_by_date.get(iso, 0) + count
    return winners_by_date


def parse_pick6_page(html: str, game_cfg: GameConfig) -> list[dict[str, Any]]:
    """
    Parse a lottopcso.com yearly history page for a pick-6 game.
    Returns a list of raw row dicts (not yet validated).
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict[str, Any]] = []

    # Build jackpot winner lookup from the 6-column winners table.
    # If the table is present (has entries), draws missing from it had 0 jackpot winners.
    jackpot_winners = _parse_winner_table(soup, game_cfg)
    has_winner_table = bool(jackpot_winners)

    # Only parse tables whose header row is exactly:
    # "Draw Date" | "Winning Numbers" | "Jackpot ..."
    # This skips the prize-claim table and the recent-winners table.
    def _is_draw_history_table(tbl) -> bool:
        header_row = tbl.find("tr")
        if not header_row:
            return False
        headers = [th.get_text(" ", strip=True).lower()
                   for th in header_row.find_all(["th", "td"])]
        return (len(headers) == 3
                and "date" in headers[0]
                and ("number" in headers[1] or "combination" in headers[1] or "winning" in headers[1]))

    for table in soup.find_all("table"):
        if not _is_draw_history_table(table):
            continue
        for tr in table.find_all("tr")[1:]:  # skip header
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if len(cells) < 3:
                continue

            raw_date = cells[0]
            raw_combo = cells[1]
            raw_jackpot = cells[2] if len(cells) > 2 else ""

            draw_date = _parse_date(raw_date)
            if draw_date is None:
                logger.warning("unparseable date: %r", raw_date)
                continue

            # Numbers: any 1-3 digit sequences separated by dashes or spaces
            nums = re.findall(r"\d{1,2}", raw_combo)
            pub_nums: list[int | None] = [int(n) for n in nums[:game_cfg.picks]] if len(nums) >= game_cfg.picks else []

            row: dict[str, Any] = {
                "draw_date": draw_date.isoformat(),
                "weekday": draw_date.strftime("%a"),
                "game": game_cfg.id,
                "draw_time": None,
                "jackpot_php": _parse_jackpot(raw_jackpot),
                "winners_jackpot": jackpot_winners.get(draw_date.isoformat(), 0 if has_winner_table else None),
                "notes": None,
            }
            for i, n in enumerate(pub_nums, 1):
                row[f"pub_{i}"] = n
            for i in range(len(pub_nums) + 1, game_cfg.picks + 1):
                row[f"pub_{i}"] = None

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# URL builders for lottopcso.com
# ---------------------------------------------------------------------------

_GAME_SLUGS: dict[str, str] = {
    "658": "6-58-lotto-result-history-and-summary",
    "655": "6-55-lotto-result-history-and-summary",
    "649": "6-49-lotto-result-history-and-summary",
    "645": "6-45-lotto-result-history-and-summary",
    "642": "6-42-lotto-result-history-and-summary",
}

BASE_URL = "https://www.lottopcso.com"


def pick6_year_url(game_id: str, year: int) -> str:
    slug = _GAME_SLUGS.get(game_id)
    if slug is None:
        raise ValueError(f"No URL slug for game {game_id!r}")
    if year == datetime.now().year:
        return f"{BASE_URL}/{slug}/"
    return f"{BASE_URL}/{slug}/{year}/"


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

async def scrape_pick6(
    game_cfg: GameConfig,
    years: list[int],
    db_conn: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch + parse pick-6 history pages for the given years.
    Optionally logs to scrape_log table if db_conn is provided.
    Returns all parsed raw rows.
    """
    all_rows: list[dict[str, Any]] = []

    for year in years:
        url = pick6_year_url(game_cfg.id, year)
        try:
            html, cache_path = await fetch_page(url)
            rows = parse_pick6_page(html, game_cfg)
            logger.info("game=%s year=%d rows=%d", game_cfg.id, year, len(rows))

            if db_conn is not None:
                db_conn.execute("""
                    INSERT INTO scrape_log (id, url, fetched_at, status, rows_parsed, cache_path)
                    VALUES (nextval('scrape_log_seq'), ?, NOW(), 200, ?, ?)
                """, [url, len(rows), str(cache_path)])

            all_rows.extend(rows)
        except Exception as exc:
            logger.error("scrape failed url=%s: %s", url, exc)
            if db_conn is not None:
                db_conn.execute("""
                    INSERT INTO scrape_log (id, url, fetched_at, status, rows_parsed, cache_path)
                    VALUES (nextval('scrape_log_seq'), ?, NOW(), 0, NULL, NULL)
                """, [url])

    return all_rows


# ---------------------------------------------------------------------------
# Official PCSO scraper — pcso.gov.ph/searchlottoresult.aspx
# ---------------------------------------------------------------------------

_PCSO_SEARCH_URL = "https://www.pcso.gov.ph/searchlottoresult.aspx"

# Maps pcso.gov.ph game name fragments (lowercase) → our game IDs
_PCSO_GAME_MAP: dict[str, str] = {
    "ultra lotto 6/58": "658",
    "grand lotto 6/55": "655",
    "superlotto 6/49": "649",
    "megalotto 6/45": "645",
    "lotto 6/42": "642",
    "6d lotto": "6d",
    "4d lotto": "4d",
    "3d lotto": "3d",
    "2d lotto": "2d",
    "ez2 lotto": "2d",
    "swertres": "3d",
}

# Draw time strings pcso.gov.ph appends to digit game names
_TIME_LABEL_MAP: dict[str, str] = {
    "2pm": "14:00",
    "5pm": "17:00",
    "9pm": "21:00",
    "11am": "11:00",
    "4pm": "16:00",
}


def _extract_draw_time(game_name_lower: str) -> str | None:
    for label, t in _TIME_LABEL_MAP.items():
        if label in game_name_lower.replace(" ", ""):
            return t
    return None


def _parse_pcso_gov_results(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict[str, Any]] = []

    results_table = None
    for tbl in soup.find_all("table"):
        ths = [th.get_text(strip=True).upper() for th in tbl.find_all("th")]
        if any("COMBINATION" in h for h in ths):
            results_table = tbl
            break

    if not results_table:
        logger.warning("pcso.gov.ph: no results table found in response")
        return rows

    for tr in results_table.find_all("tr")[1:]:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) < 4:
            continue

        game_name_raw = cells[0].strip()
        game_name_lower = game_name_raw.lower()

        game_id: str | None = None
        for key, gid in _PCSO_GAME_MAP.items():
            if key in game_name_lower:
                game_id = gid
                break
        if not game_id:
            logger.debug("pcso.gov.ph: unknown game %r — skipped", cells[0])
            continue

        game_cfg = GAMES.get(game_id)
        if not game_cfg:
            continue

        combo = cells[1].strip()
        date_str = cells[2].strip()
        jackpot_str = cells[3].strip() if len(cells) > 3 else ""
        winners_str = cells[4].strip() if len(cells) > 4 else ""

        draw_date = _parse_date(date_str)
        if not draw_date:
            logger.warning("pcso.gov.ph: unparseable date %r", date_str)
            continue

        nums = re.findall(r"\d{1,3}", combo)
        pub_nums = [int(n) for n in nums[:game_cfg.picks]] if len(nums) >= game_cfg.picks else []

        winners_jackpot: int | None = None
        if winners_str:
            try:
                winners_jackpot = int(re.sub(r"[^\d]", "", winners_str) or "0") or None
            except (ValueError, TypeError):
                pass

        row: dict[str, Any] = {
            "draw_date": draw_date.isoformat(),
            "weekday": draw_date.strftime("%a"),
            "game": game_id,
            "draw_time": _extract_draw_time(game_name_lower),
            "jackpot_php": _parse_jackpot(jackpot_str),
            "winners_jackpot": winners_jackpot,
            "notes": None,
        }
        for i, n in enumerate(pub_nums, 1):
            row[f"pub_{i}"] = n
        for i in range(len(pub_nums) + 1, game_cfg.picks + 1):
            row[f"pub_{i}"] = None

        rows.append(row)

    return rows


_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Exact ASP.NET field names discovered from the live page
_PCSO_FIELDS = {
    "from_month": "ctl00$ctl00$cphContainer$cpContent$ddlStartMonth",
    "from_day":   "ctl00$ctl00$cphContainer$cpContent$ddlStartDate",   # note: not ddlStartDay
    "from_year":  "ctl00$ctl00$cphContainer$cpContent$ddlStartYear",
    "to_month":   "ctl00$ctl00$cphContainer$cpContent$ddlEndMonth",
    "to_day":     "ctl00$ctl00$cphContainer$cpContent$ddlEndDay",
    "to_year":    "ctl00$ctl00$cphContainer$cpContent$ddlEndYear",
    "game":       "ctl00$ctl00$cphContainer$cpContent$ddlSelectGame",
    "submit":     "ctl00$ctl00$cphContainer$cpContent$btnSearch",
}


async def _fetch_pcso_gov_chunk(from_date: date, to_date: date) -> list[dict[str, Any]]:
    """
    Use curl-cffi (Chrome TLS impersonation) to bypass Cloudflare on pcso.gov.ph,
    POST the search form, and return parsed rows for one date-range chunk.
    """
    from curl_cffi.requests import AsyncSession
    from bs4 import BeautifulSoup as BS

    async with AsyncSession(impersonate="chrome124") as s:
        # GET to harvest hidden ViewState/EventValidation fields
        resp = await s.get(_PCSO_SEARCH_URL)
        resp.raise_for_status()
        soup = BS(resp.text, "lxml")
        form = soup.find("form")
        if not form:
            raise RuntimeError("pcso.gov.ph: form not found (status %d)" % resp.status_code)

        payload: dict[str, str] = {}
        for inp in form.find_all("input", type="hidden"):
            name = inp.get("name")
            if name:
                payload[name] = inp.get("value", "")

        payload[_PCSO_FIELDS["from_month"]] = _MONTH_NAMES[from_date.month]
        payload[_PCSO_FIELDS["from_day"]]   = str(from_date.day)
        payload[_PCSO_FIELDS["from_year"]]  = str(from_date.year)
        payload[_PCSO_FIELDS["to_month"]]   = _MONTH_NAMES[to_date.month]
        payload[_PCSO_FIELDS["to_day"]]     = str(to_date.day)
        payload[_PCSO_FIELDS["to_year"]]    = str(to_date.year)
        payload[_PCSO_FIELDS["game"]]       = "0"  # All Games
        payload[_PCSO_FIELDS["submit"]]     = "Search Lotto"

        logger.info("pcso.gov.ph POST %s → %s", from_date, to_date)
        await asyncio.sleep(RATE_LIMIT_S)
        resp2 = await s.post(_PCSO_SEARCH_URL, data=payload)
        resp2.raise_for_status()

    rows = _parse_pcso_gov_results(resp2.text)
    logger.info("pcso.gov.ph %s→%s: %d rows parsed", from_date, to_date, len(rows))
    return rows


async def scrape_pcso_gov(
    from_date: date,
    to_date: date,
) -> list[dict[str, Any]]:
    """
    Scrape official PCSO results from pcso.gov.ph for all games in a date range.
    Chunks into 1-month intervals to keep response sizes manageable.
    Returns raw row dicts for all games — caller groups by game before calling load_and_clean.
    """
    all_rows: list[dict[str, Any]] = []

    # Walk month by month
    current = from_date.replace(day=1)
    while current <= to_date:
        # Last day of current month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        chunk_end = min(next_month - timedelta(days=1), to_date)
        chunk_start = max(current, from_date)

        try:
            rows = await _fetch_pcso_gov_chunk(chunk_start, chunk_end)
            all_rows.extend(rows)
        except Exception as exc:
            logger.error("pcso.gov.ph chunk %s→%s failed: %s", chunk_start, chunk_end, exc)

        current = next_month

    return all_rows

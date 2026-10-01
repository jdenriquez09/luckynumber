"""CLI entry point: python -m lotto <command>"""
import asyncio
import sys
from pathlib import Path

import click

from .games import GAMES, get_game
from .scrape import scrape_pick6
from .clean import load_and_clean, completeness_report
from .report import generate_report, generate_summary
from api.db import get_connection


@click.group()
def cli() -> None:
    """PCSO Lotto data pipeline and analysis."""


@cli.command()
@click.option("--game", required=True, help="Game id, e.g. 655")
@click.option("--year", multiple=True, type=int, help="Year(s) to scrape (repeatable)")
@click.option("--from-year", "from_year", type=int, default=None, help="Start year (inclusive)")
@click.option("--to-year", "to_year", type=int, default=None, help="End year (inclusive)")
def scrape(game: str, year: tuple[int, ...], from_year: int | None, to_year: int | None) -> None:
    """Fetch and cache draw history pages."""
    game_cfg = GAMES.get(game.lower())
    if game_cfg is None:
        click.echo(f"Unknown game: {game}", err=True)
        sys.exit(1)
    if game_cfg.ordered:
        click.echo("Digit game scraping not yet implemented.", err=True)
        sys.exit(1)

    from datetime import datetime
    if year:
        years = list(year)
    elif from_year or to_year:
        start = from_year or 2016
        end = to_year or datetime.now().year
        years = list(range(start, end + 1))
    else:
        years = list(range(2021, datetime.now().year + 1))

    click.echo(f"Scraping {game_cfg.name} for years {years}")
    conn = get_connection()
    raw = asyncio.run(scrape_pick6(game_cfg, years, db_conn=conn))
    inserted, flagged = load_and_clean(raw, game_cfg, conn)
    conn.close()
    click.echo(f"Done — inserted={inserted}, flagged={flagged}")


@cli.command()
@click.option("--game", required=True, help="Game id, e.g. 655")
def report(game: str) -> None:
    """Generate Markdown report + charts for a game."""
    game_cfg = GAMES.get(game.lower())
    if game_cfg is None:
        click.echo(f"Unknown game: {game}", err=True)
        sys.exit(1)
    conn = get_connection()
    path = generate_report(game_cfg, conn)
    conn.close()
    click.echo(f"Report written to {path}")


@cli.command()
def summary() -> None:
    """Generate cross-game summary report."""
    conn = get_connection()
    path = generate_summary(conn)
    conn.close()
    click.echo(f"Summary written to {path}")


@cli.command()
@click.option("--game", default=None, help="Limit to one game id, e.g. 655 (default: all games)")
def backfill(game: str | None) -> None:
    """Compute and populate derived columns (draw_sum, odd_count, etc.) for existing rows."""
    from .migrate import run_full_migration, backfill_derived_columns, add_derived_columns

    conn = get_connection()
    try:
        added = add_derived_columns(conn)
        if added:
            click.echo(f"Added columns: {', '.join(added)}")
        result = backfill_derived_columns(conn, game=game)
        rows = result.get("rows_updated", 0)
        click.echo(f"Backfill complete — {rows} rows updated")
    finally:
        conn.close()


@cli.command()
@click.option("--game", required=True, help="Game id, e.g. 655")
@click.option("--from-year", "from_year", type=int, default=None,
              help="Start year (default: game's launch year)")
@click.option("--to-year", "to_year", type=int, default=2015,
              show_default=True, help="End year (inclusive)")
def history(game: str, from_year: int | None, to_year: int) -> None:
    """Scrape pre-2016 historical data for a pick-6 game."""
    from .migrate import scrape_historical, GAME_START_YEARS

    game_cfg = GAMES.get(game.lower())
    if game_cfg is None:
        click.echo(f"Unknown game: {game}", err=True)
        sys.exit(1)
    if game_cfg.ordered:
        click.echo("Historical scraping not supported for ordered games.", err=True)
        sys.exit(1)

    start = from_year or GAME_START_YEARS.get(game.lower(), 2010)
    click.echo(f"Scraping {game_cfg.name} history from {start} to {to_year}…")

    def _progress(year: int, inserted: int) -> None:
        click.echo(f"  {year}: {inserted} rows inserted")

    result = asyncio.run(
        scrape_historical(game.lower(), from_year=start, to_year=to_year, progress_cb=_progress)
    )
    click.echo(
        f"Done — total inserted={result['total_inserted']}, "
        f"flagged={result['total_flagged']}, "
        f"years={result['years_scraped']}"
    )


@cli.command()
@click.option("--game", required=True, help="Game id, e.g. 655")
def completeness(game: str) -> None:
    """Print completeness report for a game."""
    import json
    game_cfg = GAMES.get(game.lower())
    if game_cfg is None:
        click.echo(f"Unknown game: {game}", err=True)
        sys.exit(1)
    conn = get_connection()
    rep = completeness_report(game_cfg, conn)
    conn.close()
    click.echo(json.dumps(rep, indent=2))


@cli.command()
@click.option("--handle", default="harrychristian/pcso-lottery-results-philippines",
              show_default=True, help="Kaggle dataset handle (owner/dataset)")
@click.option("--file", "file_path", default="", show_default=True,
              help="File path within the dataset (empty = auto-detect)")
@click.option("--inspect", is_flag=True, default=False,
              help="Just print columns and sample rows; do not import")
def kaggle(handle: str, file_path: str, inspect: bool) -> None:
    """Import PCSO draw history from a Kaggle dataset."""
    from .kaggle_import import import_kaggle_dataset, inspect_kaggle_dataset
    if inspect:
        inspect_kaggle_dataset(handle, file_path)
        return
    click.echo(f"Importing from Kaggle: {handle} …")
    conn = get_connection()
    inserted, skipped = import_kaggle_dataset(handle, file_path, conn=conn)
    conn.close()
    click.echo(f"Done — inserted={inserted}, skipped/duplicate={skipped}")


@cli.command("import-csv")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, default=False,
              help="Parse only — do not write to the database")
def import_csv(file_path: str, dry_run: bool) -> None:
    """Import PCSO draw history from a local CSV file.

    Expected columns: LOTTO GAME, COMBINATIONS, DRAW DATE, JACKPOT (PHP), WINNERS
    """
    from .kaggle_import import import_csv_file
    click.echo(f"Importing from {file_path} …")
    conn = None if dry_run else get_connection()
    inserted, skipped = import_csv_file(file_path, conn=conn, dry_run=dry_run)
    if conn is not None:
        conn.close()
    click.echo(f"Done — inserted={inserted}, skipped/duplicate={skipped}")


if __name__ == "__main__":
    cli()

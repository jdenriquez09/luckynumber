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
def scrape(game: str, year: tuple[int, ...]) -> None:
    """Fetch and cache draw history pages."""
    game_cfg = GAMES.get(game.lower())
    if game_cfg is None:
        click.echo(f"Unknown game: {game}", err=True)
        sys.exit(1)
    if game_cfg.ordered:
        click.echo("Digit game scraping not yet implemented.", err=True)
        sys.exit(1)

    from datetime import datetime
    years = list(year) if year else list(range(2021, datetime.now().year + 1))
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


if __name__ == "__main__":
    cli()

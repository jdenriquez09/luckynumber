"""
Generate per-game Markdown reports + PNG figures in reports/.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .games import GameConfig, GAMES, pick6_games
from . import analysis as an
from .clean import completeness_report

REPORTS_DIR = Path("reports")
FIGS_DIR = REPORTS_DIR / "figs"


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGS_DIR.mkdir(exist_ok=True)


def _load_df(game_cfg: GameConfig, conn: Any) -> pd.DataFrame:
    num_cols = ", ".join(f"n{i}" for i in range(1, game_cfg.picks + 1))
    return conn.execute(f"""
        SELECT draw_date, weekday, game, draw_time,
               pub_1, pub_2, pub_3, pub_4, pub_5, pub_6,
               {num_cols}, jackpot_php, notes
        FROM draws WHERE game = ?
        ORDER BY draw_date
    """, [game_cfg.id]).fetchdf()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _freq_chart(result: an.AnalysisResult, game_cfg: GameConfig) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    t = result.table
    expected = t["expected"].iloc[0]
    ax.bar(t["number"], t["observed"], color="#4C72B0", alpha=0.7, label="Observed")
    ax.axhline(expected, color="red", linestyle="--", linewidth=1.2, label=f"Expected ({expected:.1f})")
    ax.set_xlabel("Number")
    ax.set_ylabel("Count")
    ax.set_title(f"{game_cfg.name} — Number Frequency")
    ax.legend()
    plt.tight_layout()
    p = FIGS_DIR / f"{game_cfg.id}_freq.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _gap_chart(result: an.AnalysisResult, game_cfg: GameConfig) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    t = result.table.head(20)
    ax.barh(t["number"].astype(str), t["current_gap"], color="#DD8452")
    ax.set_xlabel("Draws since last appearance")
    ax.set_title(f"{game_cfg.name} — Current Gap (top 20 longest absent)")
    plt.tight_layout()
    p = FIGS_DIR / f"{game_cfg.id}_gap.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def generate_report(game_cfg: GameConfig, conn: Any) -> Path:
    _ensure_dirs()
    df = _load_df(game_cfg, conn)
    coverage = completeness_report(game_cfg, conn)

    lines: list[str] = [
        f"# {game_cfg.name} Statistical Report",
        f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "\n---\n",
        "## Data Coverage",
        f"- **First draw:** {coverage['first']}",
        f"- **Last draw:** {coverage['last']}",
        f"- **Total draws:** {coverage['count']} (expected {coverage.get('expected', '?')})",
        f"- **Calendar gaps:** {len(coverage.get('gaps', []))}",
    ]
    if coverage.get("gaps"):
        lines += ["\n<details><summary>Gap dates</summary>\n\n" +
                  "\n".join(f"- {g}" for g in coverage["gaps"]) + "\n\n</details>"]

    lines += ["\n---\n", "## Statistical Findings\n"]

    if not df.empty and df[[f"n{i}" for i in range(1, game_cfg.picks + 1)]].notna().any().any():
        if not game_cfg.ordered:
            # Pick-6 analyses
            for func, chart_fn in [
                (an.freq_chi_square, _freq_chart),
                (an.distribution_tests, None),
                (an.sequence_tests, None),
                (an.gap_analysis, _gap_chart),
                (an.monte_carlo, None),
                (an.jackpot_dynamics, None),
                (an.date_effects, None),
            ]:
                try:
                    result = func(df, game_cfg)  # type: ignore[call-arg]
                    lines += [f"### {result.name.replace('_', ' ').title()}", ""]
                    lines += [result.conclusion, ""]
                    if not result.table.empty:
                        lines += [result.table.to_markdown(index=False), ""]
                    if chart_fn is not None:
                        p = chart_fn(result, game_cfg)
                        lines += [f"![chart]({p.relative_to(REPORTS_DIR)})", ""]
                except Exception as exc:
                    lines += [f"### Error in {func.__name__}: {exc}", ""]
        else:
            for func in [an.digit_uniformity, an.digit_repeat_rate]:
                try:
                    result = func(df, game_cfg)
                    lines += [f"### {result.name.replace('_', ' ').title()}", ""]
                    lines += [result.conclusion, ""]
                    if not result.table.empty:
                        lines += [result.table.to_markdown(index=False), ""]
                except Exception as exc:
                    lines += [f"### Error in {func.__name__}: {exc}", ""]
    else:
        lines += ["_No draw data loaded yet. Run the scraper first._"]

    lines += [
        "\n---\n",
        "## Limitations",
        "- Results test for statistical consistency with fair random draws.",
        "- They do NOT identify which numbers will be drawn next.",
        "- Lotto draws are designed to be independent and uniformly random.",
        "- A significant p-value with many tests is expected by chance alone.",
        "- Data coverage may be incomplete; check the gaps section above.",
    ]

    report_path = REPORTS_DIR / f"{game_cfg.id}_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_summary(conn: Any) -> Path:
    """Summary report comparing all games."""
    _ensure_dirs()
    rows_by_game = conn.execute("""
        SELECT game, COUNT(*) AS draws,
               MIN(draw_date) AS first, MAX(draw_date) AS last
        FROM draws GROUP BY game
    """).fetchdf()

    lines = [
        "# PCSO Lotto — Summary Report",
        f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        "## Data Coverage by Game\n",
        rows_by_game.to_markdown(index=False),
        "\n---\n",
        "## Notes",
        "- Statistical tests are described per-game in their individual reports.",
        "- No predictions are made. This analysis tests fairness, not forecasts.",
    ]

    p = REPORTS_DIR / "summary_report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p

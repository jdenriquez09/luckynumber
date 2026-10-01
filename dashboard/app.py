"""
Streamlit multi-page analyst dashboard.

Pages:
  1. Raw Data Viewer
  2. Frequency Analysis
  3. Statistical Tests
  4. Completeness Report
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api.db import get_connection, compare_with_actual
from src.lotto.games import GAMES
import src.lotto.analysis as an
from src.lotto.clean import completeness_report

st.set_page_config(
    page_title="PCSO Lotto Analysis",
    page_icon="🎱",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Shared helpers — open a fresh connection per query so the file is never
# held open permanently (allows the CLI scraper to write concurrently).
# ---------------------------------------------------------------------------

def _query(sql: str, params: list | None = None) -> pd.DataFrame:
    conn = get_connection()
    try:
        result = conn.execute(sql, params or []).fetchdf()
    finally:
        conn.close()
    return result


_INT_COLS = (
    [f"pub_{i}" for i in range(1, 7)]
    + [f"n{i}" for i in range(1, 7)]
    + ["winners_jackpot", "winners_5match", "winners_4match", "winners_3match",
       "draw_sum", "odd_count", "high_count", "consecutive_cnt", "number_spread"]
)

@st.cache_data(ttl=30)
def load_draws(game_id: str) -> pd.DataFrame:
    game_cfg = GAMES[game_id]
    num_cols = ", ".join(f"n{i}" for i in range(1, game_cfg.picks + 1))
    df = _query(f"""
        SELECT draw_date, weekday, game, draw_time,
               pub_1, pub_2, pub_3, pub_4, pub_5, pub_6,
               {num_cols}, jackpot_php, notes,
               draw_sum, odd_count, high_count, consecutive_cnt, number_spread,
               is_rollover, covid_gap,
               winners_jackpot, winners_5match, winners_4match, winners_3match
        FROM draws WHERE game = ?
        ORDER BY draw_date DESC
    """, [game_id])
    # Cast number columns from float64 (pandas default with NULLs) to Int64
    for col in _INT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


@st.cache_data(max_entries=32, show_spinner=False)
def _prob_generation(_df: pd.DataFrame, dataset_hash: str, game_id: str,
                     window: int | None, draw_time: str, n: int,
                     mode: str, model_version: str) -> dict:
    """Cache per-slot tickets/metrics; explicit snapshot and version key _df."""
    from src.lotto.lean_ml import generate_ml_combinations, _prepare_draws
    from src.lotto.predict import random_combinations

    cfg = GAMES[game_id]
    if mode == "ML-assisted":
        return generate_ml_combinations(
            _df, cfg, n=n, window=window, draw_time=draw_time, seed=42,
        )
    return {
        "method": "uniform_fallback",
        "combinations": random_combinations(cfg, n=n, seed=42),
        "scores": [], "feature_weights": {}, "training_draws": 0,
        "reason": "Uniform random selected; no model was trained or evaluated.",
        "evaluation": {
            "status": "not_requested", "test_draws": 0,
            "usable_draws": len(_prepare_draws(_df, cfg, window, draw_time)),
            "model_brier": None, "baseline_brier": None, "brier_skill": None,
            "improvement_ci95": [], "evidence_of_improvement": False,
        },
    }


def game_selector(label: str = "Game") -> str:
    options = {g.id: g.name for g in GAMES.values()}
    return st.selectbox(label, list(options.keys()),
                        format_func=lambda k: options[k])


def _show_interp(template_msg: str, key_facts: str, game_name: str,
                 chart_name: str, cache_key: str) -> None:
    """Show chart interpretation — AI text if cached for current data, else template + AI button.

    The actual session_state key is versioned with the latest draw date so that
    when new draws are scraped the old AI text is automatically replaced by the
    template (and a fresh AI call can be triggered).
    """
    from src.lotto.azure_ai import AzureAIClient as _AIC
    _ai = _AIC()
    requested = _ai.is_configured() and st.button(
        "Generate / Regenerate AI explanation", key=f"regen_{cache_key}"
    )
    if requested:
        st.session_state.pop(cache_key, None)
    if cache_key in st.session_state:
        st.info(st.session_state[cache_key])
    elif requested:
        with st.spinner("AI is analysing this chart…"):
            try:
                explanation = _ai.explain_chart(game_name, chart_name, key_facts)
                st.session_state[cache_key] = explanation
                st.rerun()
            except Exception as _e:
                st.info(template_msg)
                st.error(f"AI error: {_e}")
    else:
        st.info(template_msg)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = [
    "Raw Data", "Frequency Analysis", "Statistical Tests", "Completeness",
    "Probability & AI", "Backtest", "Advanced Stats", "Combo Tools",
    "Gambler's Edge", "Data Quality", "Saved Combos",
]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")

# ---- Quick Update button ----
def _get_last_dates() -> dict[str, str]:
    """Return {game_id: last_draw_date_iso} for all games that have data."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT game, MAX(draw_date) AS last_date
            FROM draws
            GROUP BY game
        """).fetchdf()
    finally:
        conn.close()
    if rows.empty:
        return {}
    return dict(zip(rows["game"].tolist(), rows["last_date"].astype(str).tolist()))


import datetime as _dt_mod

_last_dates = _get_last_dates()
_today_upd = _dt_mod.date.today()

if _last_dates:
    _overall_last = max(_last_dates.values())
    _last_dt = _dt_mod.date.fromisoformat(_overall_last)
    _days_behind = (_today_upd - _last_dt).days
    if _days_behind > 0:
        st.sidebar.warning(f"Data is **{_days_behind} day(s)** behind (last: {_overall_last})")
    else:
        st.sidebar.success(f"Data is up to date (last: {_overall_last})")
else:
    _days_behind = 9999
    st.sidebar.info("No data yet — use Load Data below.")

_update_disabled = _days_behind <= 0

# Auto-update: trigger scrape automatically when data is stale
_auto_update = _days_behind > 0 and "_auto_updated" not in st.session_state

if _auto_update or st.sidebar.button(
    "🔄 Update Data (all games)",
    disabled=_update_disabled,
    help="Scrape new draws from the day after the last recorded draw up to today.",
    use_container_width=True,
):
    import asyncio
    from collections import defaultdict
    from src.lotto.scrape import scrape_pcso_gov
    from src.lotto.clean import load_and_clean as _lac

    # Start from the day after the latest draw already in the DB,
    # or from 30 days ago if there's no data at all
    if _last_dates:
        _from_date = _dt_mod.date.fromisoformat(max(_last_dates.values())) + _dt_mod.timedelta(days=1)
    else:
        _from_date = _today_upd - _dt_mod.timedelta(days=30)

    if _from_date > _today_upd:
        st.session_state["_auto_updated"] = True
        st.sidebar.success("Already up to date — nothing to fetch.")
    else:
        _prog = st.sidebar.empty()
        _prog.info(f"Fetching {_from_date} → {_today_upd}…")
        try:
            _raw_all = asyncio.run(scrape_pcso_gov(_from_date, _today_upd))
            _by_game: dict = defaultdict(list)
            for _r in _raw_all:
                _by_game[_r["game"]].append(_r)

            _conn_w = get_connection()
            _total_ins = _total_flg = 0
            _game_summary: list[str] = []
            for _gid, _grows in _by_game.items():
                if _gid in GAMES:
                    _ins, _flg = _lac(_grows, GAMES[_gid], _conn_w)
                    _total_ins += _ins
                    _total_flg += _flg
                    if _ins:
                        _game_summary.append(f"**{GAMES[_gid].name}**: +{_ins} draw(s)")
            _conn_w.close()
            st.cache_data.clear()
            # Clear ALL cached AI results and ML models
            for _sk in [k for k in st.session_state if k.startswith(("ai_", "ml_"))]:
                del st.session_state[_sk]
            _prog.empty()

            st.session_state["_auto_updated"] = True
            if _total_ins:
                st.sidebar.success(
                    f"✅ Updated! **{_total_ins} new draw(s)** added across "
                    f"{len(_game_summary)} game(s).\n\n" + "\n\n".join(_game_summary)
                )
            else:
                st.sidebar.info("No new draws found — data is already current.")
        except Exception as _upd_exc:
            st.session_state["_auto_updated"] = True
            _prog.empty()
            st.sidebar.error(f"Update failed: {_upd_exc}")

# Manual AI + ML cache clear
_n_cached = sum(1 for k in st.session_state if k.startswith(("ai_", "ml_")))
if _n_cached:
    if st.sidebar.button(
        f"🧹 Clear AI/ML Cache ({_n_cached} stored)",
        help="Remove all cached AI results, chart interpretations, and ML models.",
        use_container_width=True,
    ):
        for _k in [k for k in st.session_state if k.startswith(("ai_", "ml_"))]:
            del st.session_state[_k]
        st.rerun()

st.sidebar.markdown("---")

# ---- In-app scraper (manual range) ----
with st.sidebar.expander("Load Data (custom range)"):
    scrape_source = st.radio(
        "Source",
        ["pcso.gov.ph (official)", "lottopcso.com"],
        key="scrape_source",
        help="pcso.gov.ph is the official source and loads all games at once.",
    )

    if scrape_source == "pcso.gov.ph (official)":
        st.caption("pcso.gov.ph has data from 2016 onwards. For earlier years use **Load Historical Data** below.")
        _today = _dt_mod.date.today()
        _jan1 = _dt_mod.date(2016, 1, 1)
        pcso_from = st.date_input("From", value=_jan1,
                                   min_value=_dt_mod.date(2016, 1, 1),
                                   max_value=_today, key="pcso_from")
        pcso_to = st.date_input("To", value=_today,
                                 min_value=_dt_mod.date(2016, 1, 1),
                                 max_value=_today, key="pcso_to")

        if st.button("Scrape (all games)", key="scrape_pcso_btn"):
            import asyncio
            from collections import defaultdict
            from src.lotto.scrape import scrape_pcso_gov
            from src.lotto.clean import load_and_clean
            with st.spinner(f"Scraping pcso.gov.ph {pcso_from} → {pcso_to}…"):
                raw_all = asyncio.run(scrape_pcso_gov(pcso_from, pcso_to))
                by_game: dict = defaultdict(list)
                for _r in raw_all:
                    by_game[_r["game"]].append(_r)
                conn_w = get_connection()
                total_ins = total_flg = 0
                for gid, game_rows in by_game.items():
                    if gid in GAMES:
                        ins, flg = load_and_clean(game_rows, GAMES[gid], conn_w)
                        total_ins += ins
                        total_flg += flg
                conn_w.close()
                st.cache_data.clear()
            st.success(
                f"Done — {total_ins} rows across {len(by_game)} game(s) loaded, "
                f"{total_flg} flagged."
            )

    else:  # lottopcso.com
        scrape_game = st.selectbox("Game", list(GAMES.keys()),
                                   format_func=lambda k: GAMES[k].name,
                                   key="scrape_game_sel")
        current_year = _dt_mod.datetime.now().year
        scrape_years = st.multiselect(
            "Years", list(range(1995, current_year + 1)),
            default=[current_year], key="scrape_years_sel"
        )
        if st.button("Scrape now", key="scrape_btn"):
            import asyncio
            from src.lotto.scrape import scrape_pick6
            from src.lotto.clean import load_and_clean
            game_cfg_s = GAMES[scrape_game]
            with st.spinner(f"Scraping {game_cfg_s.name} for {scrape_years}…"):
                conn_w = get_connection()
                raw = asyncio.run(scrape_pick6(game_cfg_s, scrape_years, db_conn=conn_w))
                ins, flg = load_and_clean(raw, game_cfg_s, conn_w)
                conn_w.close()
                st.cache_data.clear()
            st.success(f"Done — {ins} rows loaded, {flg} flagged.")

st.sidebar.markdown("---")

# ---- Historical data loader (any year range) ----
with st.sidebar.expander("Load Historical Data"):
    st.caption(
        "Scrape any year range for pick-6 games via lottopcso.com. "
        "Covers data from game inception to the present."
    )
    from src.lotto.migrate import GAME_START_YEARS as _GAME_START_YEARS
    import asyncio as _asyncio
    from src.lotto.migrate import scrape_historical as _scrape_hist

    _hist_pick6_games = {gid: g for gid, g in GAMES.items() if not g.ordered}

    _hist_scope = st.radio(
        "Scope", ["All pick-6 games", "Single game"], key="hist_scope_sel"
    )

    if _hist_scope == "Single game":
        _hist_game = st.selectbox(
            "Game", list(_hist_pick6_games.keys()),
            format_func=lambda k: _hist_pick6_games[k].name,
            key="hist_game_sel",
        )
        _hist_games_to_run = {_hist_game: _hist_pick6_games[_hist_game]}
        _default_start = _GAME_START_YEARS.get(_hist_game, 2010)
    else:
        _hist_game = None
        _hist_games_to_run = _hist_pick6_games
        _default_start = min(_GAME_START_YEARS.get(gid, 2010) for gid in _hist_pick6_games)

    _hist_cur_year = _dt_mod.datetime.now().year
    _hist_from = st.number_input(
        "From year", min_value=1990, max_value=_hist_cur_year,
        value=int(_default_start), step=1, key="hist_from_yr"
    )
    _hist_to = st.number_input(
        "To year", min_value=1990, max_value=_hist_cur_year,
        value=_hist_cur_year, step=1, key="hist_to_yr"
    )

    if st.button("Load Historical Draws", key="hist_scrape_btn"):
        _hist_status = st.empty()
        _hist_grand_total = 0
        _hist_games_done = 0

        for _hgid, _hgcfg in _hist_games_to_run.items():
            _game_launch = _GAME_START_YEARS.get(_hgid, int(_hist_from))
            _eff_from = max(int(_hist_from), _game_launch)
            if _eff_from > int(_hist_to):
                continue
            _hist_status.info(
                f"Scraping {_hgcfg.name} ({_eff_from}–{int(_hist_to)})… "
                f"({_hist_games_done}/{len(_hist_games_to_run)} done)"
            )
            try:
                _hist_result = _asyncio.run(
                    _scrape_hist(_hgid, from_year=_eff_from, to_year=int(_hist_to))
                )
                _hist_grand_total += _hist_result["total_inserted"]
                _hist_games_done += 1
            except Exception as _he:
                st.warning(f"{_hgcfg.name}: {_he}")
                _hist_games_done += 1

        st.cache_data.clear()
        _hist_status.empty()
        st.success(
            f"Done — **{_hist_grand_total} rows** loaded "
            f"across {_hist_games_done} game(s)."
        )

# ---- Kaggle dataset importer ----
with st.sidebar.expander("Import from Kaggle"):
    st.caption(
        "Load pre-2016 PCSO draw data from a Kaggle dataset. "
        "Requires a Kaggle account and API key (~/.kaggle/kaggle.json)."
    )
    _kg_handle = st.text_input(
        "Dataset handle",
        value="harrychristian/pcso-lottery-results-philippines",
        key="kg_handle",
    )
    _kg_file = st.text_input(
        "File path (leave blank to auto-detect)",
        value="",
        key="kg_file",
    )
    _kg_col1, _kg_col2 = st.columns(2)
    if _kg_col1.button("🔍 Inspect columns", key="kg_inspect_btn"):
        try:
            from src.lotto.kaggle_import import inspect_kaggle_dataset as _kginspect
            with st.spinner("Loading dataset preview…"):
                _kg_df = _kginspect(_kg_handle, _kg_file)
            st.write("**Shape:**", _kg_df.shape)
            st.write("**Columns:**", list(_kg_df.columns))
            st.dataframe(_kg_df.head(5), use_container_width=True)
        except Exception as _kge:
            st.error(f"Inspect failed: {_kge}")

    if _kg_col2.button("⬇️ Import", key="kg_import_btn"):
        try:
            from src.lotto.kaggle_import import import_kaggle_dataset as _kgimport
            with st.spinner("Downloading and importing Kaggle dataset…"):
                _conn_kg = get_connection()
                _kg_ins, _kg_skip = _kgimport(_kg_handle, _kg_file, conn=_conn_kg)
                _conn_kg.close()
                st.cache_data.clear()
            st.success(
                f"✅ **{_kg_ins} rows** inserted, {_kg_skip} skipped (duplicates/invalid)."
            )
        except Exception as _kge:
            st.error(f"Import failed: {_kge}")

# ---- CSV file uploader ----
with st.sidebar.expander("Import from CSV"):
    st.caption(
        "Upload a CSV file with columns: "
        "LOTTO GAME, COMBINATIONS, DRAW DATE, JACKPOT (PHP), WINNERS"
    )
    _csv_file = st.file_uploader("Choose CSV file", type=["csv"], key="csv_upload")
    if _csv_file is not None:
        if st.button("⬆️ Import CSV", key="csv_import_btn"):
            import tempfile, os
            try:
                from src.lotto.kaggle_import import import_csv_file as _csv_import
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as _tmp:
                    _tmp.write(_csv_file.getvalue())
                    _tmp_path = _tmp.name
                with st.spinner("Importing CSV…"):
                    _conn_csv = get_connection()
                    _csv_ins, _csv_skip = _csv_import(_tmp_path, conn=_conn_csv)
                    _conn_csv.close()
                    st.cache_data.clear()
                os.unlink(_tmp_path)
                st.success(
                    f"✅ **{_csv_ins} rows** inserted, {_csv_skip} skipped (duplicates/invalid)."
                )
            except Exception as _csve:
                st.error(f"Import failed: {_csve}")

# ---- Backfill derived columns ----
with st.sidebar.expander("Compute Derived Stats"):
    st.caption(
        "Populate draw_sum, odd_count, high_count, etc. for all existing rows. "
        "Run once after loading data or after a schema upgrade."
    )
    _bf_game_opts = ["All games"] + [f"{gid} — {g.name}" for gid, g in GAMES.items()]
    _bf_selection = st.selectbox("Scope", _bf_game_opts, key="bf_game_sel")
    if st.button("Run Backfill", key="bf_run_btn"):
        from src.lotto.migrate import add_derived_columns as _add_cols, backfill_derived_columns as _bfd
        _bf_game_id = None if _bf_selection == "All games" else _bf_selection.split(" — ")[0]
        with st.spinner("Backfilling…"):
            try:
                _conn_bf = get_connection()
                _added = _add_cols(_conn_bf)
                _bf_res = _bfd(_conn_bf, game=_bf_game_id)
                _conn_bf.close()
                st.cache_data.clear()
                _msg = f"✅ **{_bf_res.get('rows_updated', 0)} rows** updated."
                if _added:
                    _msg += f" New columns: {', '.join(_added)}."
                st.success(_msg)
            except Exception as _bfe:
                st.error(f"Backfill failed: {_bfe}")

st.sidebar.markdown("---")
st.sidebar.caption(
    "This dashboard tests whether PCSO draws are statistically consistent "
    "with a fair random process. It does NOT predict future draws."
)

# ---------------------------------------------------------------------------
# Page 1: Raw Data Viewer
# ---------------------------------------------------------------------------

if page == "Raw Data":
    st.title("Raw Draw Data")
    game_id = game_selector()
    game_cfg = GAMES[game_id]

    _rd_min = _dt_mod.date(1995, 1, 1)
    _rd_max = _dt_mod.date.today()
    col1, col2 = st.columns(2)
    with col1:
        from_date = st.date_input("From date", value=None,
                                   min_value=_rd_min, max_value=_rd_max)
    with col2:
        to_date = st.date_input("To date", value=None,
                                 min_value=_rd_min, max_value=_rd_max)

    df = load_draws(game_id)

    if from_date:
        df = df[pd.to_datetime(df["draw_date"]) >= pd.Timestamp(from_date)]
    if to_date:
        df = df[pd.to_datetime(df["draw_date"]) <= pd.Timestamp(to_date)]

    flagged = df[df["notes"].notna() & (df["notes"] != "")]
    st.metric("Total draws", len(df))
    st.metric("Flagged rows", len(flagged))

    show_flagged = st.checkbox("Show flagged rows only")
    display_df = (flagged if show_flagged else df).copy()

    # Pre-format jackpot as string so commas render correctly (printf has no comma specifier)
    display_df["jackpot_php"] = display_df["jackpot_php"].apply(
        lambda x: f"{x:,.2f}" if pd.notna(x) else ""
    )

    # Fill NULL winner counts with 0 for consistent display
    for _wc in ["winners_jackpot", "winners_5match", "winners_4match", "winners_3match"]:
        if _wc in display_df.columns:
            display_df[_wc] = display_df[_wc].fillna(0)

    def _style_rows(row: pd.Series) -> list[str]:
        has_note = bool(row.get("notes")) and str(row.get("notes", "")).strip() not in ("", "None", "nan")
        if has_note:
            return ["background-color: #b45309; color: #ffffff; font-weight: 500"] * len(row)
        return [""] * len(row)

    # Column display config: integers show without decimals, dates stay as strings
    col_cfg: dict = {
        "draw_date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "jackpot_php": st.column_config.TextColumn(
            "Jackpot (PHP)", help="Jackpot prize in Philippine Peso"
        ),
        "notes": st.column_config.TextColumn("Notes", width="large"),
    }
    for c in _INT_COLS:
        if c in display_df.columns:
            col_cfg[c] = st.column_config.NumberColumn(c, format="%d", width="small")

    _MAX_STYLED_CELLS = 262144
    _use_style = (len(display_df) * len(display_df.columns)) <= _MAX_STYLED_CELLS
    _render_df = display_df.style.apply(_style_rows, axis=1) if _use_style else display_df
    if not _use_style and len(flagged):
        st.info("Row highlighting disabled for large datasets (too many cells for Pandas Styler). Use 'Show flagged rows only' to inspect flagged draws.")

    st.dataframe(
        _render_df,
        use_container_width=True,
        height=520,
        column_config=col_cfg,
    )

    if len(flagged):
        with st.expander(f"⚠ {len(flagged)} flagged rows — what does this mean?", expanded=True):
            st.markdown("""
**Flagged rows** (amber) are draws where the source data has a quality issue:

| Note text | Meaning |
|---|---|
| `draw_date … is Thu but 658 draws on Tue,Fri,Sun` | Date from the website falls on the **wrong weekday** — possible typo on the source site |
| `missing numbers` | The website row had no winning combination (suspended draw or missing data) |
| `numbers out of range` | One or more numbers outside the valid pool for this game |
| `jackpot non-positive` | Jackpot value was zero or negative — blanked to avoid polluting analysis |

Numbers and jackpot are **blanked** for flagged rows. They are stored but excluded from statistical analysis.
            """)


    st.download_button(
        "Download CSV",
        display_df.to_csv(index=False).encode(),
        f"{game_id}_draws.csv",
        "text/csv",
    )

# ---------------------------------------------------------------------------
# Page 2: Frequency Analysis
# ---------------------------------------------------------------------------

elif page == "Frequency Analysis":
    st.title("Number Frequency")
    game_id = game_selector()
    game_cfg = GAMES[game_id]
    df = load_draws(game_id)
    if df.empty:
        st.warning("No data loaded. Run the scraper first.")
    else:
        result = an.freq_chi_square(df.sort_values("draw_date"), game_cfg)
        t = result.table

        expected = t["expected"].iloc[0]
        fig = go.Figure()
        fig.add_bar(x=t["number"], y=t["observed"], name="Observed", marker_color="#4C72B0")
        fig.add_scatter(
            x=t["number"], y=[expected] * len(t),
            mode="lines", name=f"Expected ({expected:.1f})",
            line=dict(color="red", dash="dash"),
        )
        fig.update_layout(
            title=f"{game_cfg.name} — Number Frequency",
            xaxis_title="Number",
            yaxis_title="Count",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Human-readable interpretation
        top_n = t.loc[t["observed"].idxmax()]
        bot_n = t.loc[t["observed"].idxmin()]
        outliers_hi = t[t["deviation"] > 2] if "deviation" in t.columns else pd.DataFrame()
        outliers_lo = t[t["deviation"] < -2] if "deviation" in t.columns else pd.DataFrame()
        n_total = int(t["observed"].sum())
        p_val = result.meta.get("p") if result.meta else None
        chi2_val = result.meta.get("chi2") if result.meta else None

        msg = (
            f"**What this chart shows:** Each blue bar is the total number of times that number appeared "
            f"across all {n_total:,} draw slots. The red dashed line is the expected count if every number "
            f"had a perfectly equal chance of being drawn ({expected:.0f} times).\n\n"
            f"**Most drawn:** Number **{int(top_n['number'])}** appeared **{int(top_n['observed'])}** times "
            f"({(top_n['observed'] - expected) / expected * 100:+.1f}% above expected).\n\n"
            f"**Least drawn:** Number **{int(bot_n['number'])}** appeared **{int(bot_n['observed'])}** times "
            f"({(bot_n['observed'] - expected) / expected * 100:+.1f}% vs expected).\n\n"
        )
        if not outliers_hi.empty:
            msg += f"**Unusually high (z > 2):** Numbers {outliers_hi['number'].tolist()} have appeared significantly more than chance alone would predict.\n\n"
        if not outliers_lo.empty:
            msg += f"**Unusually low (z < −2):** Numbers {outliers_lo['number'].tolist()} have appeared significantly less than expected.\n\n"
        if p_val is not None and chi2_val is not None:
            if p_val >= 0.05:
                msg += (
                    f"**Overall verdict:** The chi-square test (χ²={chi2_val:.2f}, p={p_val:.4f}) finds "
                    f"**no statistically significant departure from a fair draw** — the differences you see "
                    f"are within the normal range of random variation."
                )
            else:
                msg += (
                    f"**Overall verdict:** The chi-square test (χ²={chi2_val:.2f}, p={p_val:.4f}) detects "
                    f"**a statistically significant imbalance** — some numbers appear more or less often than "
                    f"a truly fair draw would produce."
                )
        msg += "\n\n_Even when a number appears more often historically, this does NOT make it more likely to appear in future draws. Each draw is an independent random event._"

        # Build key facts for AI
        _freq_facts = (
            f"Total draw slots: {n_total:,}\n"
            f"Expected count per number (fair draw): {expected:.0f}\n"
            f"Most drawn: number {int(top_n['number'])} appeared {int(top_n['observed'])} times "
            f"({(top_n['observed']-expected)/expected*100:+.1f}% vs expected)\n"
            f"Least drawn: number {int(bot_n['number'])} appeared {int(bot_n['observed'])} times "
            f"({(bot_n['observed']-expected)/expected*100:+.1f}% vs expected)\n"
            + (f"Numbers well above expected: {outliers_hi['number'].tolist()}\n" if not outliers_hi.empty else "")
            + (f"Numbers well below expected: {outliers_lo['number'].tolist()}\n" if not outliers_lo.empty else "")
            + f"Overall verdict: {'fair draw — no significant imbalance' if p_val and p_val >= 0.05 else 'significant imbalance detected'}"
            + (f" (p={p_val:.4f})" if p_val else "")
        )
        _show_interp(msg, _freq_facts, game_cfg.name, "Number Frequency",
                     f"ai_interp_{game_id}_freq_analysis")

        with st.expander("Technical result & full frequency table"):
            st.caption(result.conclusion)
            st.dataframe(t, use_container_width=True)

# ---------------------------------------------------------------------------
# Page 3: Statistical Tests
# ---------------------------------------------------------------------------

elif page == "Statistical Tests":
    st.title("Statistical Tests")
    game_id = game_selector()
    game_cfg = GAMES[game_id]
    df = load_draws(game_id)
    if df.empty:
        st.warning("No data loaded. Run the scraper first.")
    else:
        df_sorted = df.sort_values("draw_date")

        if game_cfg.ordered:
            test_map = {
                "Digit Uniformity": an.digit_uniformity,
                "Digit Repeat Rate": an.digit_repeat_rate,
            }
            if len(game_cfg.draw_times) > 1:
                test_map["Draw-Time Comparison"] = an.draw_time_comparison
        else:
            test_map = {
                "Frequency Chi-Square": an.freq_chi_square,
                "Distribution Tests": an.distribution_tests,
                "Pair Co-occurrence (FDR)": an.pair_cooccurrence,
                "Gap Analysis": an.gap_analysis,
                "Sequence Tests": an.sequence_tests,
                "Monte Carlo": an.monte_carlo,
                "Jackpot Dynamics": an.jackpot_dynamics,
                "Date Effects": an.date_effects,
            }

        test_name = st.selectbox("Select test", list(test_map.keys()))
        func = test_map[test_name]

        with st.spinner("Running analysis…"):
            try:
                result = func(df_sorted, game_cfg)
                st.success(result.conclusion)
                if not result.table.empty:
                    st.dataframe(result.table, use_container_width=True)

                # ---- Per-test visualizations ----
                t = result.table

                if test_name == "Frequency Chi-Square" and "number" in t.columns:
                    fig = go.Figure()
                    fig.add_bar(x=t["number"], y=t["observed"], name="Observed",
                                marker_color="#4C72B0")
                    fig.add_bar(x=t["number"],
                                y=[t["expected"].iloc[0]] * len(t),
                                name="Expected", marker_color="#55A868", opacity=0.6)
                    outliers = t[t["deviation"].abs() > 2]
                    if not outliers.empty:
                        fig.add_scatter(x=outliers["number"], y=outliers["observed"],
                                        mode="markers", marker=dict(color="red", size=8),
                                        name="|z|>2 outlier")
                    fig.update_layout(barmode="overlay", xaxis_title="Number",
                                      yaxis_title="Count", title="Observed vs Expected Frequency")
                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation
                    top_n = t.loc[t["observed"].idxmax()]
                    bot_n = t.loc[t["observed"].idxmin()]
                    hi = t[t["deviation"] > 2]
                    lo = t[t["deviation"] < -2]
                    exp = t["expected"].iloc[0]
                    msg = (
                        f"**What this chart shows:** Each blue bar is how many times a number actually appeared "
                        f"across all draws. The green overlay is the expected count if every number had an equal "
                        f"chance of being drawn. Red dots highlight numbers that deviate significantly.\n\n"
                        f"**Most drawn:** Number **{int(top_n['number'])}** appeared **{int(top_n['observed'])}** times "
                        f"({(top_n['observed']-exp)/exp*100:+.1f}% vs expected {exp:.0f}).\n\n"
                        f"**Least drawn:** Number **{int(bot_n['number'])}** appeared only **{int(bot_n['observed'])}** times "
                        f"({(bot_n['observed']-exp)/exp*100:+.1f}% vs expected).\n\n"
                    )
                    if not hi.empty:
                        msg += f"**Above normal (z > 2):** Numbers {hi['number'].tolist()} have appeared more often than statistics alone would predict.\n\n"
                    if not lo.empty:
                        msg += f"**Below normal (z < −2):** Numbers {lo['number'].tolist()} have appeared less often than expected.\n\n"
                    msg += "_Remember: in a truly random lottery, some numbers will always appear more or less than average — this is normal variation, not a pattern you can exploit._"
                    _facts = (
                        f"Total draws: {int(t['observed'].sum())}\n"
                        f"Expected per number: {exp:.0f}\n"
                        f"Most drawn: {int(top_n['number'])} ({int(top_n['observed'])} times, {(top_n['observed']-exp)/exp*100:+.1f}%)\n"
                        f"Least drawn: {int(bot_n['number'])} ({int(bot_n['observed'])} times, {(bot_n['observed']-exp)/exp*100:+.1f}%)\n"
                        + (f"Numbers much higher than expected: {hi['number'].tolist()}\n" if not hi.empty else "")
                        + (f"Numbers much lower than expected: {lo['number'].tolist()}\n" if not lo.empty else "")
                        + f"Overall: {'fair draw' if result.meta.get('p',1) >= 0.05 else 'significant imbalance'}"
                    )
                    _show_interp(msg, _facts, game_cfg.name, "Frequency Chi-Square",
                                 f"ai_interp_{game_id}_freq_chi")

                elif test_name == "Gap Analysis" and "number" in t.columns:
                    exp_gap = t["expected_gap"].iloc[0] if not t.empty else 0
                    colors = [
                        "#f59e0b" if cg > 2 * exp_gap else "#4C72B0"
                        for cg in t["current_gap"]
                    ]
                    fig = go.Figure()
                    fig.add_scatter(x=t["number"], y=t["mean_gap"], mode="markers+lines",
                                    name="Mean gap (average interval between appearances)",
                                    marker_color="#4C72B0")
                    fig.add_scatter(x=t["number"], y=t["current_gap"], mode="markers",
                                    name="Current gap (draws since last seen)",
                                    marker=dict(color=colors, size=8, symbol="diamond"))
                    fig.add_hline(y=exp_gap, line_dash="dash", line_color="#55A868",
                                  annotation_text=f"Expected gap ({exp_gap:.1f} draws)")
                    fig.update_layout(xaxis_title="Number", yaxis_title="Number of draws",
                                      title="Gap Analysis: How Often Each Number Appears",
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation
                    overdue = t[t["current_gap"] > 2 * exp_gap].sort_values("current_gap", ascending=False)
                    recent = t.nsmallest(3, "current_gap")
                    long_avg = t.nlargest(3, "mean_gap")
                    msg = (
                        f"**What this chart shows:** The blue line is each number's average interval between "
                        f"appearances (how many draws typically pass before it shows up again). "
                        f"The diamond markers show how many draws have passed since each number last appeared. "
                        f"The green dashed line is the theoretical average: every number should appear roughly "
                        f"every **{exp_gap:.1f} draws** if the lottery is fair.\n\n"
                    )
                    if not overdue.empty:
                        rows = [f"**{int(r['number'])}** ({int(r['current_gap'])} draws — {r['current_gap']/exp_gap:.1f}× overdue)" for _, r in overdue.head(5).iterrows()]
                        msg += f"**Amber diamonds (long absence):** {', '.join(rows)}. These numbers haven't appeared in a long time compared to their own historical average.\n\n"
                    msg += (
                        f"**Recently seen:** Numbers {recent['number'].tolist()} appeared within the last few draws.\n\n"
                        f"**Widest historical gaps (slowest numbers):** {long_avg['number'].tolist()} have the longest average intervals.\n\n"
                        "_Important: A number that hasn't appeared for a long time is NOT more likely to appear next. "
                        "Each draw is independent — the lottery has no memory._"
                    )
                    _facts = (
                        f"Expected gap between appearances: {exp_gap:.1f} draws\n"
                        + (f"Numbers long overdue (gap > 2x expected): {[int(r['number']) for _,r in overdue.head(5).iterrows()]} with current gaps {[int(r['current_gap']) for _,r in overdue.head(5).iterrows()]}\n" if not overdue.empty else "No numbers dramatically overdue.\n")
                        + f"Most recently seen: {recent['number'].tolist()}\n"
                        + f"Historically slowest numbers (widest average gaps): {long_avg['number'].tolist()}"
                    )
                    _show_interp(msg, _facts, game_cfg.name, "Gap Analysis",
                                 f"ai_interp_{game_id}_gap")

                elif test_name == "Pair Co-occurrence (FDR)" and "p_fdr" in t.columns:
                    sig_rows = t[t.get("significant", pd.Series([False]*len(t)))] if "significant" in t.columns else pd.DataFrame()
                    top_chart = sig_rows.head(20) if not sig_rows.empty else t.head(20)
                    top_chart = top_chart.copy()
                    top_chart["pair"] = top_chart["num_a"].astype(str) + " & " + top_chart["num_b"].astype(str)
                    top_chart["color"] = (top_chart["significant"] if "significant" in top_chart.columns else pd.Series([False]*len(top_chart))).map(
                        {True: "#e05252", False: "#4C72B0"})
                    fig = go.Figure(go.Bar(
                        x=top_chart["observed"], y=top_chart["pair"], orientation="h",
                        marker_color=top_chart["color"].tolist(),
                        hovertemplate="Pair: %{y}<br>Appeared together: %{x} times<extra></extra>",
                    ))
                    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                                      xaxis_title="Times appeared together",
                                      title="Most Significant Number Pairs (🔴 red = statistically unusual co-occurrence)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation
                    sig_count = len(sig_rows)
                    total_pairs = len(t)
                    exp_co = t["expected"].iloc[0] if "expected" in t.columns and not t.empty else None
                    msg = (
                        f"**What this chart shows:** Every possible pair of numbers was checked to see how often "
                        f"they appeared in the same draw. Red bars are pairs whose co-occurrence rate is statistically "
                        f"unusual (after a strict false-discovery-rate correction that accounts for testing {total_pairs:,} pairs at once). "
                        f"Blue bars are within the normal range.\n\n"
                    )
                    if sig_count > 0 and not top_chart.empty:
                        top_pair_row = top_chart.iloc[-1]
                        msg += (
                            f"**{sig_count} unusual pair(s) found** out of {total_pairs:,} tested. "
                            f"The most notable: **{top_pair_row['pair']}** appeared together "
                            f"**{int(top_pair_row['observed'])} times**"
                            + (f" vs an expected {exp_co:.1f}" if exp_co else "") + ".\n\n"
                        )
                    else:
                        msg += f"**No unusual pairs** found after statistical correction. All {total_pairs:,} pairs appear at roughly the frequency you'd expect from pure chance.\n\n"
                    msg += "_Even if two numbers appear together often in history, this tells you nothing about whether they'll appear together in the next draw._"
                    _facts = (
                        f"Total number pairs tested: {total_pairs:,}\n"
                        f"Unusual pairs found (after correction): {sig_count}\n"
                        + (f"Most notable pair: {top_chart.iloc[-1]['pair']} appeared together {int(top_chart.iloc[-1]['observed'])} times" + (f" vs expected {exp_co:.1f}" if exp_co else "") if sig_count > 0 and not top_chart.empty else "All pairs appear at normal expected frequency.")
                    )
                    _show_interp(msg, _facts, game_cfg.name, "Pair Co-occurrence",
                                 f"ai_interp_{game_id}_pairs")

                elif test_name == "Date Effects" and "factor" in t.columns:
                    df_sorted["draw_date"] = pd.to_datetime(df_sorted["draw_date"])
                    df_sorted["month"] = df_sorted["draw_date"].dt.month_name().str[:3]
                    num_cols_de = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
                    df_sorted["draw_sum"] = df_sorted[num_cols_de].sum(axis=1)
                    c1_de, c2_de = st.columns(2)
                    with c1_de:
                        wd_grp = df_sorted.groupby("weekday")["draw_sum"].agg(["mean", "std"]).reset_index()
                        fig1 = go.Figure(go.Bar(
                            x=wd_grp["weekday"], y=wd_grp["mean"],
                            error_y=dict(type="data", array=wd_grp["std"].tolist()),
                            marker_color="#4C72B0",
                        ))
                        fig1.update_layout(title="Average Draw Sum by Day of Week",
                                           yaxis_title="Average sum of drawn numbers",
                                           xaxis_title="Day")
                        st.plotly_chart(fig1, use_container_width=True)
                    with c2_de:
                        mo_grp = df_sorted.groupby("month")["draw_sum"].agg(["mean", "std"]).reset_index()
                        fig2 = go.Figure(go.Bar(
                            x=mo_grp["month"], y=mo_grp["mean"],
                            error_y=dict(type="data", array=mo_grp["std"].tolist()),
                            marker_color="#55A868",
                        ))
                        fig2.update_layout(title="Average Draw Sum by Month",
                                           yaxis_title="Average sum of drawn numbers",
                                           xaxis_title="Month")
                        st.plotly_chart(fig2, use_container_width=True)

                    # Interpretation
                    wd_high = wd_grp.loc[wd_grp["mean"].idxmax()]
                    wd_low = wd_grp.loc[wd_grp["mean"].idxmin()]
                    mo_high = mo_grp.loc[mo_grp["mean"].idxmax()]
                    mo_low = mo_grp.loc[mo_grp["mean"].idxmin()]
                    wd_range = wd_grp["mean"].max() - wd_grp["mean"].min()
                    mo_range = mo_grp["mean"].max() - mo_grp["mean"].min()
                    wd_sig = t[t["factor"] == "weekday"]["p_value"].values[0] if "weekday" in t["factor"].values else None
                    mo_sig = t[t["factor"] == "month"]["p_value"].values[0] if "month" in t["factor"].values else None
                    msg = (
                        f"**What this chart shows:** The average of all numbers drawn in a single draw is called the 'draw sum'. "
                        f"These charts check whether draws on certain days or months tend to produce higher or lower sums — "
                        f"which would hint at a weekday or seasonal bias. The error bars show the spread (±1 standard deviation).\n\n"
                        f"**By day of week:** {wd_high['weekday']} draws have the highest average sum ({wd_high['mean']:.1f}), "
                        f"while {wd_low['weekday']} has the lowest ({wd_low['mean']:.1f}). "
                        f"The difference is {wd_range:.1f} points "
                        + (f"— {'statistically significant (p={wd_sig:.4f})' if wd_sig and wd_sig < 0.05 else 'within normal random variation (p={:.4f})'.format(wd_sig) if wd_sig else ''}") + ".\n\n"
                        f"**By month:** {mo_high['month']} draws tend to have the highest sums ({mo_high['mean']:.1f}), "
                        f"{mo_low['month']} the lowest ({mo_low['mean']:.1f}). Difference: {mo_range:.1f} points "
                        + (f"— {'statistically significant (p={:.4f})'.format(mo_sig) if mo_sig and mo_sig < 0.05 else 'within normal random variation'}") + ".\n\n"
                        "_Even when ANOVA finds a significant difference, the variation is tiny compared to the randomness within each draw. No day or month gives you a meaningful edge._"
                    )
                    _facts = (
                        f"Day with highest average draw sum: {wd_high['weekday']} ({wd_high['mean']:.1f})\n"
                        f"Day with lowest average draw sum: {wd_low['weekday']} ({wd_low['mean']:.1f})\n"
                        f"Difference across weekdays: {wd_range:.1f} points — {'significant' if wd_sig and wd_sig < 0.05 else 'within normal variation'}\n"
                        f"Month with highest average: {mo_high['month']} ({mo_high['mean']:.1f})\n"
                        f"Month with lowest average: {mo_low['month']} ({mo_low['mean']:.1f})\n"
                        f"Difference across months: {mo_range:.1f} points — {'significant' if mo_sig and mo_sig < 0.05 else 'within normal variation'}"
                    )
                    _show_interp(msg, _facts, game_cfg.name, "Date Effects",
                                 f"ai_interp_{game_id}_date")

                elif test_name == "Digit Uniformity" and "position" in t.columns:
                    num_cols_du = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
                    pool_du = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
                    tabs_du = st.tabs([f"Position {p}" for p in range(1, game_cfg.picks + 1)])
                    for pos_idx, tab_du in enumerate(tabs_du, 1):
                        with tab_du:
                            col_du = num_cols_du[pos_idx - 1]
                            counts_du = df_sorted[col_du].dropna().astype(int).value_counts()
                            counts_du = counts_du.reindex(pool_du, fill_value=0)
                            exp_du = len(df_sorted) / len(pool_du)
                            colors_du = [
                                "#22c55e" if c > exp_du + 2*(exp_du**0.5) else
                                ("#e05252" if c < exp_du - 2*(exp_du**0.5) else "#4C72B0")
                                for c in counts_du.values
                            ]
                            fig_du = go.Figure(go.Bar(
                                x=[str(d) for d in pool_du], y=counts_du.values,
                                marker_color=colors_du,
                                hovertemplate="Digit %{x}: appeared %{y} times<extra></extra>",
                            ))
                            fig_du.add_hline(y=exp_du, line_dash="dash", line_color="#f59e0b",
                                             annotation_text=f"Expected ({exp_du:.0f})")
                            fig_du.update_layout(xaxis_title="Digit", yaxis_title="Times drawn",
                                                 title=f"Position {pos_idx} — How Often Each Digit Appears Here")
                            st.plotly_chart(fig_du, use_container_width=True)

                            top_d = counts_du.idxmax(); bot_d = counts_du.idxmin()
                            above = [d for d, c in zip(pool_du, counts_du.values) if c > exp_du + 2*(exp_du**0.5)]
                            below = [d for d, c in zip(pool_du, counts_du.values) if c < exp_du - 2*(exp_du**0.5)]
                            row = t[t["position"] == pos_idx]
                            p_val = row["p_value"].values[0] if not row.empty else None
                            msg = (
                                f"**Position {pos_idx}** — Each bar shows how many times that digit appeared in draw position {pos_idx}. "
                                f"The orange dashed line is the expected count if every digit were equally likely ({exp_du:.0f} times). "
                                f"Green = above normal, red = below normal.\n\n"
                                f"**Most common here:** Digit **{top_d}** ({int(counts_du[top_d])} draws, "
                                f"{(counts_du[top_d]-exp_du)/exp_du*100:+.1f}% vs expected).\n\n"
                                f"**Least common here:** Digit **{bot_d}** ({int(counts_du[bot_d])} draws, "
                                f"{(counts_du[bot_d]-exp_du)/exp_du*100:+.1f}% vs expected).\n\n"
                            )
                            if above:
                                msg += f"**Statistically high (green):** Digits {above} appear more than chance alone explains.\n\n"
                            if below:
                                msg += f"**Statistically low (red):** Digits {below} appear less than chance alone explains.\n\n"
                            if not above and not below:
                                msg += "**All digits are within the normal range** — no position bias detected here.\n\n"
                            if p_val is not None:
                                msg += f"Chi-square test: p = {p_val:.4f} — {'significant departure from uniform' if p_val < 0.05 else 'no significant departure from uniform'}.\n\n"
                            msg += "_Ordered games (3D, 2D, 4D) draw digits independently at each position. A digit that frequently appears in Position 1 is not more likely there in the next draw._"
                            st.info(msg)  # template only inside tabs

                    # Combined AI explanation covering all positions
                    _num_cols_du2 = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
                    _pool_du2 = list(range(game_cfg.low_boundary, game_cfg.pool + 1))
                    _exp_du2 = len(df_sorted) / len(_pool_du2)
                    _du_parts = []
                    for _pos in range(1, game_cfg.picks + 1):
                        _col2 = _num_cols_du2[_pos - 1]
                        _cnts = df_sorted[_col2].dropna().astype(int).value_counts().reindex(_pool_du2, fill_value=0)
                        _hi2 = [d for d, c in zip(_pool_du2, _cnts.values) if c > _exp_du2 + 2*(_exp_du2**0.5)]
                        _lo2 = [d for d, c in zip(_pool_du2, _cnts.values) if c < _exp_du2 - 2*(_exp_du2**0.5)]
                        _row2 = t[t["position"] == _pos]
                        _pv2 = _row2["p_value"].values[0] if not _row2.empty else None
                        _du_parts.append(
                            f"Position {_pos}: most common digit={_cnts.idxmax()} ({int(_cnts.max())}x), "
                            f"least common={_cnts.idxmin()} ({int(_cnts.min())}x)"
                            + (f", unusually high: {_hi2}" if _hi2 else "")
                            + (f", unusually low: {_lo2}" if _lo2 else "")
                            + (f", fairness test: {'not fair' if _pv2 and _pv2 < 0.05 else 'fair'} (p={_pv2:.4f})" if _pv2 else "")
                        )
                    _facts_du = f"Expected count per digit per position: {_exp_du2:.0f}\n" + "\n".join(_du_parts)
                    _du_summary = (
                        f"This test checks whether each digit (0–{game_cfg.pool}) appears equally often "
                        f"in each position across all draws. "
                        f"Expected: each digit should appear roughly {_exp_du2:.0f} times per position. "
                        f"Any bias could hint that certain positions favour certain digits."
                    )
                    _show_interp(_du_summary, _facts_du, game_cfg.name, "Digit Uniformity",
                                 f"ai_interp_{game_id}_digit_unif")

                elif test_name == "Draw-Time Comparison" and "number" in t.columns:
                    time_cols = [c for c in t.columns if c != "number"]
                    fig_dtc = go.Figure()
                    colors_dtc = ["#4C72B0", "#55A868", "#e05252"]
                    for i, tc in enumerate(time_cols):
                        fig_dtc.add_bar(x=t["number"], y=t[tc], name=tc,
                                        marker_color=colors_dtc[i % len(colors_dtc)])
                    fig_dtc.update_layout(barmode="group", xaxis_title="Number / Digit",
                                          yaxis_title="Times drawn",
                                          title="Number Frequency Compared Across Draw Times",
                                          legend_title="Draw time")
                    st.plotly_chart(fig_dtc, use_container_width=True)

                    # Interpretation
                    p_val = result.meta.get("p")
                    chi2 = result.meta.get("chi2")
                    n_draws = result.meta.get("n")
                    # Find digit with biggest variation across draw times
                    if len(time_cols) >= 2:
                        t2 = t.set_index("number")[time_cols]
                        t2["range"] = t2.max(axis=1) - t2.min(axis=1)
                        most_var = t2["range"].idxmax()
                        least_var = t2["range"].idxmin()
                        row_var = t2.loc[most_var]
                        msg = (
                            f"**What this chart shows:** For each number/digit, grouped bars compare how many times "
                            f"it appeared in the **{time_cols[0]}**, **{time_cols[1]}**"
                            + (f", and **{time_cols[2]}**" if len(time_cols) > 2 else "") +
                            f" draws. If the bars for each number are roughly the same height, the three draw times "
                            f"are statistically indistinguishable.\n\n"
                            f"**Statistical test:** χ² = {chi2:.2f}, p = {p_val:.4f} "
                            f"({'The draw times have significantly different frequency profiles' if p_val and p_val < 0.05 else 'No significant difference between draw times'}).\n\n"
                            f"**Most variable number:** **{most_var}** shows the biggest gap between draw times "
                            f"({', '.join(f'{tc}: {int(row_var[tc])}' for tc in time_cols)}).\n\n"
                            f"**Most consistent:** **{least_var}** appears at almost the same rate across all three draw times.\n\n"
                        )
                    else:
                        msg = f"Chi-square p = {p_val:.4f}.\n\n"
                    msg += "_If draw times are statistically different, it suggests each time slot should be analysed separately — which is why this platform generates separate combinations per draw time._"
                    _facts_dtc = (
                        f"Draw times compared: {time_cols}\n"
                        + (
                            f"Fairness test: {'significant difference between draw times' if p_val and p_val < 0.05 else 'no significant difference'} (p={p_val:.4f})\n"
                            f"Most variable number: {most_var} ({', '.join(f'{tc}: {int(row_var[tc])}' for tc in time_cols)})\n"
                            f"Most consistent number: {least_var}"
                            if len(time_cols) >= 2 else f"p={p_val:.4f}"
                        )
                    )
                    _show_interp(msg, _facts_dtc, game_cfg.name, "Draw-Time Comparison",
                                 f"ai_interp_{game_id}_drawtime")

                elif test_name == "Digit Repeat Rate":
                    n_draws_dr = result.meta.get("n", len(df_sorted))
                    obs = t.set_index("metric")["value"].to_dict() if not t.empty else {}
                    obs_rate = obs.get("observed_adjacent_repeat_rate", None)
                    exp_rate = obs.get("expected_rate", None)
                    t_stat_dr = obs.get("t_stat", None)
                    p_dr = obs.get("p_value", None)
                    msg = (
                        f"**What this measures:** In ordered games (3D, 2D, 4D), 'adjacent repeat' means the same digit "
                        f"appearing in two consecutive positions of the same draw — for example, drawing 3-3-7 in 3D. "
                        f"This test checks whether repeats happen more or less often than pure chance.\n\n"
                    )
                    if obs_rate is not None and exp_rate is not None:
                        diff_pct = (obs_rate - exp_rate) / exp_rate * 100
                        msg += (
                            f"**Observed repeat rate:** {obs_rate:.4f} ({obs_rate*100:.2f}% of adjacent position pairs)\n\n"
                            f"**Expected by chance:** {exp_rate:.4f} ({exp_rate*100:.2f}%)\n\n"
                            f"**Difference:** {diff_pct:+.1f}% — digits repeat in adjacent positions "
                            + ("more" if obs_rate > exp_rate else "less") + " than pure chance predicts.\n\n"
                        )
                    if p_dr is not None:
                        msg += f"**Statistical test (t-test):** p = {p_dr:.4f} — {'significant departure from expected' if p_dr < 0.05 else 'within normal random variation'}.\n\n"
                    msg += "_Even if adjacent repeats are unusual, this cannot tell you which draw will have a repeat next._"
                    _facts_dr = (
                        (f"Observed adjacent repeat rate: {obs_rate*100:.2f}%\n"
                         f"Expected by chance: {exp_rate*100:.2f}%\n"
                         f"Digits repeat {'more' if obs_rate and exp_rate and obs_rate > exp_rate else 'less'} than expected\n"
                         if obs_rate is not None and exp_rate is not None else "")
                        + (f"Statistical test: {'significant' if p_dr and p_dr < 0.05 else 'not significant'} (p={p_dr:.4f})" if p_dr is not None else "")
                    )
                    _show_interp(msg, _facts_dr, game_cfg.name, "Digit Repeat Rate",
                                 f"ai_interp_{game_id}_repeat")

                elif test_name == "Distribution Tests" and not t.empty:
                    msg_lines = [
                        "**What this measures:** Four different ways to check if each draw looks like a fair random pick:\n",
                    ]
                    for _, row_dt in t.iterrows():
                        name = str(row_dt["test"])
                        p = row_dt.get("p_value")
                        obs_m = row_dt.get("observed_mean")
                        exp_m = row_dt.get("expected_mean")
                        if name == "draw_sum":
                            msg_lines.append(
                                f"- **Draw sum:** The numbers in each draw add up to an average of **{obs_m}** "
                                f"(expected {exp_m}). "
                                + (f"p = {p:.4f} — {'unusually high or low' if p < 0.05 else 'consistent with a fair draw'}." if p else "")
                            )
                        elif name == "odd_count_distribution":
                            msg_lines.append(
                                f"- **Odd/even balance:** On average **{obs_m}** odd numbers per draw "
                                f"(expected {exp_m} in a fair draw). "
                                + (f"p = {p:.4f} — {'significant imbalance' if p < 0.05 else 'balanced'}." if p else "")
                            )
                        elif name == "low_count_distribution":
                            msg_lines.append(
                                f"- **Low/high balance:** On average **{obs_m}** numbers from the lower half of the pool per draw "
                                f"(expected {exp_m}). "
                                + (f"p = {p:.4f} — {'biased toward one half' if p < 0.05 else 'evenly distributed'}." if p else "")
                            )
                        elif name == "consecutive_pairs_mean":
                            msg_lines.append(
                                f"- **Consecutive pairs:** Draws contain an average of **{obs_m}** pairs of consecutive numbers "
                                f"(e.g. 14 & 15 in the same draw). No formal test — purely descriptive."
                            )
                    msg_lines.append("\n_No single test finding should be used as a basis for selecting numbers. These check overall draw fairness, not individual number prediction._")
                    _tmpl_dist = "\n".join(msg_lines)
                    _facts_dist = "\n".join(
                        f"{r['test']}: observed={r.get('observed_mean')}, expected={r.get('expected_mean')}, "
                        f"result={'significant' if r.get('p_value') and r['p_value'] < 0.05 else 'normal'}"
                        + (f" (p={r['p_value']:.4f})" if r.get('p_value') else "")
                        for _, r in t.iterrows()
                    )
                    _show_interp(_tmpl_dist, _facts_dist, game_cfg.name, "Distribution Tests",
                                 f"ai_interp_{game_id}_dist")

                elif test_name == "Sequence Tests" and not t.empty:
                    msg_lines = ["**What this measures:** Whether draws are truly independent — i.e., knowing the last draw gives no information about the next one.\n"]
                    for _, row_st in t.iterrows():
                        name = row_st["test"]
                        stat = row_st.get("statistic")
                        p = row_st.get("p_value")
                        if name == "wald_wolfowitz_runs":
                            msg_lines.append(
                                f"- **Runs test:** Checks if draw sums alternate up/down in a pattern. "
                                f"z = {stat:.3f}, p = {p:.4f} — "
                                + ("some non-random sequencing detected." if p < 0.05 else "no sequential pattern found.")
                            )
                        elif name == "lag1_autocorr_sum":
                            msg_lines.append(
                                f"- **Autocorrelation:** Does a high draw sum today predict a high sum tomorrow? "
                                f"Correlation = {stat:.4f}, p = {p:.4f} — "
                                + ("significant correlation — successive draws are not fully independent." if p < 0.05 else "no meaningful correlation between successive draws.")
                            )
                        elif name == "lag1_set_overlap":
                            msg_lines.append(
                                f"- **Number repeat draw-to-draw:** On average {stat:.2f} numbers repeat from one draw to the next. "
                                f"p = {p:.4f} — "
                                + ("significantly more or fewer repeats than expected." if p < 0.05 else "within the expected range for independent draws.")
                            )
                    msg_lines.append("\n_If any test is significant, apply Bonferroni correction (α ÷ 3 ≈ 0.017) before concluding non-randomness._")
                    _tmpl_seq = "\n".join(msg_lines)
                    _facts_seq = "\n".join(
                        f"{r['test']}: statistic={r.get('statistic'):.4f}, "
                        f"result={'sequential pattern detected' if r.get('p_value') and r['p_value'] < 0.05 else 'no pattern'}"
                        + (f" (p={r['p_value']:.4f})" if r.get('p_value') else "")
                        for _, r in t.iterrows() if r.get('statistic') is not None
                    )
                    _show_interp(_tmpl_seq, _facts_seq, game_cfg.name, "Sequence Tests",
                                 f"ai_interp_{game_id}_seq")

                elif test_name == "Monte Carlo" and not t.empty:
                    vals = t.set_index("metric")["value"].to_dict()
                    real_mean = vals.get("real_mean_sum")
                    sim_mean = vals.get("sim_mean_sum")
                    percentile = vals.get("percentile")
                    msg = (
                        f"**What this measures:** {result.meta.get('n_sim', 10000):,} fake lottery histories were simulated "
                        f"using the same pool and pick count as {game_cfg.name}. Each simulated draw picked numbers "
                        f"completely at random. The real historical data is then compared to this simulated baseline.\n\n"
                        f"**Real average draw sum:** {real_mean} &nbsp;|&nbsp; **Simulated average:** {sim_mean}\n\n"
                        f"**Percentile:** The real data sits at the **{percentile:.1f}th percentile** of {result.meta.get('n_sim', 10000):,} simulated fair histories. "
                    )
                    if percentile is not None:
                        if 2.5 <= percentile <= 97.5:
                            msg += "This is **well within the expected range** — the historical draws look statistically indistinguishable from a perfectly fair lottery."
                        elif percentile < 2.5:
                            msg += "This is in the **lower tail** — real draws have produced unusually low sums compared to what a perfectly fair lottery would generate."
                        else:
                            msg += "This is in the **upper tail** — real draws have produced unusually high sums compared to simulation."
                    msg += "\n\n_The Monte Carlo test only checks the mean draw sum. Many other aspects of the draw (individual number frequencies, pairs, etc.) are tested separately._"
                    _facts_mc = (
                        f"Simulations run: {result.meta.get('n_sim', 10000):,}\n"
                        f"Real historical average draw sum: {real_mean}\n"
                        f"Simulated fair lottery average: {sim_mean}\n"
                        f"Percentile of real data: {percentile:.1f}th\n"
                        f"Verdict: {'looks like a fair lottery' if percentile is not None and 2.5 <= percentile <= 97.5 else 'unusual compared to fair simulation'}"
                    )
                    _show_interp(msg, _facts_mc, game_cfg.name, "Monte Carlo",
                                 f"ai_interp_{game_id}_mc")

                elif test_name == "Jackpot Dynamics" and not t.empty:
                    vals = t.set_index("metric")["value"].to_dict()
                    n_wins = int(vals.get("jackpot_wins", 0))
                    win_rate = vals.get("winner_rate_pct", 0)
                    mean_roll = vals.get("mean_rollover_draws", 0)
                    total_d = int(vals.get("total_draws", 0))
                    from math import comb as _comb
                    pool_size_jd = game_cfg.pool - game_cfg.low_boundary + 1
                    theo_prob = 1 / _comb(pool_size_jd, game_cfg.picks) * 100
                    msg = (
                        f"**What this measures:** How often someone wins the jackpot and how long winning streaks typically are.\n\n"
                        f"**{n_wins} jackpot wins** detected across {total_d} draws ({win_rate:.2f}% of draws had a winner).\n\n"
                        f"**Average jackpot rollover:** {mean_roll:.1f} draws between wins — meaning the jackpot grew through "
                        f"an average of **{mean_roll:.0f} consecutive draws** before someone won.\n\n"
                        f"**Theoretical win probability per ticket:** 1 in {_comb(pool_size_jd, game_cfg.picks):,} "
                        f"({theo_prob:.6f}%). The observed win rate ({win_rate:.2f}%) reflects multiple tickets sold per draw, not cheating.\n\n"
                        f"_Jackpot size does not change your odds. Buying more tickets is the only way to improve your chances._"
                    )
                    _facts_jp = (
                        f"Total draws: {total_d}\n"
                        f"Jackpot wins recorded: {n_wins} ({win_rate:.2f}% of draws)\n"
                        f"Average draws between wins (rollover length): {mean_roll:.1f}\n"
                        f"Theoretical chance of winning per ticket: 1 in {_comb(pool_size_jd, game_cfg.picks):,}"
                    )
                    _show_interp(msg, _facts_jp, game_cfg.name, "Jackpot Dynamics",
                                 f"ai_interp_{game_id}_jackpot")

                with st.expander("Raw meta"):
                    st.json(result.meta)
            except Exception as exc:
                st.error(f"Analysis error: {exc}")

        # ---- Frequency Trends ----
        if not game_cfg.ordered:
            st.subheader("Frequency Trends")
            st.caption("Rolling frequency of each number over a sliding window of draws.")

            trend_window = st.select_slider(
                "Rolling window (draws)", options=[30, 50, 100, 200], value=50,
                key="trend_window"
            )
            with st.spinner("Computing rolling frequencies…"):
                try:
                    trend_df = an.rolling_frequency(df_sorted, game_cfg, window=trend_window)
                    if not trend_df.empty:
                        # heat as reference for top/cold numbers
                        from src.lotto.predict import number_heat_map as _nhm
                        heat_ref = _nhm(df_sorted, game_cfg)
                        top_hot = heat_ref.nlargest(10, "composite_score")["number"].tolist()
                        top_cold = heat_ref.nsmallest(10, "composite_score")["number"].tolist()

                        trend_filter = st.radio(
                            "Show numbers",
                            ["Top 10 hot", "Top 10 cold", "Custom"],
                            horizontal=True, key="trend_filter"
                        )
                        if trend_filter == "Top 10 hot":
                            show_nums = top_hot
                        elif trend_filter == "Top 10 cold":
                            show_nums = top_cold
                        else:
                            show_nums = st.multiselect(
                                "Pick numbers",
                                list(range(game_cfg.low_boundary, game_cfg.pool + 1)),
                                default=top_hot[:5],
                                key="trend_custom"
                            )

                        subset = trend_df[trend_df["number"].isin(show_nums)]
                        fig_trend = go.Figure()
                        for num in show_nums:
                            ndata = subset[subset["number"] == num].sort_values("draw_date")
                            if not ndata.empty:
                                fig_trend.add_scatter(
                                    x=ndata["draw_date"], y=ndata["rolling_count"],
                                    mode="lines", name=str(num),
                                )
                        fig_trend.update_layout(
                            xaxis_title="Draw date",
                            yaxis_title=f"Appearances in last {trend_window} draws",
                            title=f"Rolling Frequency (window={trend_window})",
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)

                        # Interpretation
                        if not subset.empty and show_nums:
                            # Find which number rose/fell the most over the window
                            rising = {}
                            falling = {}
                            for num in show_nums:
                                ndata = subset[subset["number"] == num].sort_values("draw_date")
                                if len(ndata) >= 2:
                                    first_half = ndata.head(len(ndata)//2)["rolling_count"].mean()
                                    second_half = ndata.tail(len(ndata)//2)["rolling_count"].mean()
                                    delta = second_half - first_half
                                    if delta > 0:
                                        rising[num] = delta
                                    elif delta < 0:
                                        falling[num] = delta
                            top_rising = sorted(rising, key=rising.get, reverse=True)[:3]
                            top_falling = sorted(falling, key=falling.get)[:3]
                            msg = (
                                f"**What this chart shows:** Each line tracks how many times a number appeared "
                                f"across the last **{trend_window} draws** — a rolling window that slides forward "
                                f"draw by draw. A rising line means a number has been appearing more recently; "
                                f"a falling line means it's been less frequent lately.\n\n"
                            )
                            if top_rising:
                                msg += f"**Trending up (appeared more recently):** Numbers {top_rising} have risen in frequency over the displayed period.\n\n"
                            if top_falling:
                                msg += f"**Trending down (appeared less recently):** Numbers {top_falling} have declined in frequency.\n\n"
                            if not top_rising and not top_falling:
                                msg += "**All displayed numbers are trending roughly flat** — no meaningful rise or decline in this window.\n\n"
                            msg += (
                                "_A rising trend does NOT mean a number will continue rising. "
                                "Each draw is independent — the trend is a description of the past, not a forecast._"
                            )
                            _facts_trend = (
                                f"Rolling window: {trend_window} draws\n"
                                f"Numbers shown: {show_nums}\n"
                                + (f"Trending up recently: {top_rising}\n" if top_rising else "No numbers clearly trending up.\n")
                                + (f"Trending down recently: {top_falling}\n" if top_falling else "No numbers clearly trending down.\n")
                                + ("All numbers roughly flat." if not top_rising and not top_falling else "")
                            )
                            _show_interp(msg, _facts_trend, game_cfg.name, "Frequency Trends",
                                         f"ai_interp_{game_id}_trends_{trend_window}")
                except Exception as exc:
                    st.error(f"Trend error: {exc}")

# ---------------------------------------------------------------------------
# Page 4: Completeness Report
# ---------------------------------------------------------------------------

elif page == "Completeness":
    st.title("Data Completeness")
    game_id = game_selector()
    game_cfg = GAMES[game_id]

    conn = get_connection()
    report = completeness_report(game_cfg, conn)
    conn.close()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Draws loaded", report["count"])
    c2.metric("Expected", report.get("expected", "—"))
    c3.metric("First draw", report.get("first") or "—")
    c4.metric("Last draw", report.get("last") or "—")

    gaps = report.get("gaps", [])
    st.metric("Calendar gaps", len(gaps))

    if gaps:
        st.subheader("Unexplained gaps")
        st.caption("These are scheduled draw dates with no data. May be holidays or missing scrape coverage.")
        gap_df = pd.DataFrame({"date": pd.to_datetime(gaps)})
        gap_df["weekday"] = gap_df["date"].dt.strftime("%a")
        gap_df["year"] = gap_df["date"].dt.year
        st.dataframe(gap_df, use_container_width=True)

        year_counts = gap_df["year"].value_counts().sort_index()
        fig = px.bar(year_counts, title="Gaps by Year")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No calendar gaps detected for loaded date range.")

# ---------------------------------------------------------------------------
# Page 5: Probability & AI
# ---------------------------------------------------------------------------

elif page == "Probability & AI":
    from src.lotto.predict import (
        number_heat_map, DISCLAIMER
    )
    import src.lotto.lean_ml as _lean_ml
    from src.lotto.azure_ai import AzureAIClient
    import src.lotto.analysis as _an

    _TIME_LABELS = {"14:00": "2:00 PM", "17:00": "5:00 PM", "21:00": "9:00 PM"}

    st.title("Probability & AI Analysis")

    # --- Mandatory disclaimer ---
    st.warning(
        "**Statistical patterns only — NOT predictions.**  \n"
        + DISCLAIMER,
        icon="⚠️",
    )

    game_id = game_selector()
    game_cfg = GAMES[game_id]
    df = load_draws(game_id).sort_values(["draw_date", "draw_time"])

    if df.empty:
        st.warning("No data loaded. Use the Load Data panel in the sidebar first.")
        st.stop()

    _prob_latest_date = str(df["draw_date"].max())[:10]

    is_multi_draw = len(game_cfg.draw_times) > 1

    window_choice = st.radio(
        "Analysis window",
        ["All draws", "Last 90 draws", "Last 30 draws"],
        horizontal=True,
        key="prob_window",
    )
    window_map = {"All draws": None, "Last 90 draws": 90, "Last 30 draws": 30}
    window = window_map[window_choice]

    # lean_ml exposes no version constant; hash its source to version both caches.
    import hashlib as _hashlib
    import inspect as _inspect
    _LEAN_ML_VERSION = _hashlib.sha256(
        _inspect.getsource(_lean_ml).encode()
    ).hexdigest()[:16]
    _data_hash = _hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:16]

    # --- Number Heat Map ---
    st.subheader("Number Heat Map")
    st.caption(
        "**Composite score** = blend of recent frequency and 'overdue' gap. "
        "Higher = appeared more often recently AND/OR hasn't appeared in a while. "
        "This is a historical measurement, not a forecast."
    )

    from src.lotto.predict import position_heat_map as _pos_heat_map

    def _draw_heat_chart(heat_df, label=None):
        if heat_df.empty:
            return
        if label:
            st.markdown(f"##### {label}")
        colors = [
            "#22c55e" if s > 0.65 else ("#3b82f6" if s < 0.35 else "#6b7280")
            for s in heat_df["composite_score"]
        ]
        fig = go.Figure(go.Bar(
            x=heat_df["composite_score"],
            y=heat_df["number"].astype(str),
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "Number: %{y}<br>"
                "Composite score: %{x:.3f}<br>"
                "Recent draws: %{customdata[0]}<br>"
                "Current gap: %{customdata[1]} draws<extra></extra>"
            ),
            customdata=heat_df[["recent_draws", "current_gap"]].values,
        ))
        fig.update_layout(
            height=max(300, len(heat_df) * 14),
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Composite Score",
            yaxis_title="Number",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    def _draw_position_charts(pos_heat: dict, draw_time_label: str = ""):
        if not pos_heat:
            st.info("No data for this draw time.")
            return

        # Color legend (shown once above the tabs)
        st.markdown(
            "🟢 **Dark green** — significantly above expected (z > +2) &nbsp;|&nbsp; "
            "🟩 **Light green** — slightly above expected (z +1 to +2) &nbsp;|&nbsp; "
            "🔵 **Blue** — within normal range (|z| ≤ 1) &nbsp;|&nbsp; "
            "🟠 **Amber** — slightly below expected (z −1 to −2) &nbsp;|&nbsp; "
            "🔴 **Red** — significantly below expected (z < −2)"
        )

        tabs_p = st.tabs([f"Position {p}" for p in sorted(pos_heat.keys())])
        for tab, pos in zip(tabs_p, sorted(pos_heat.keys())):
            with tab:
                ph = pos_heat[pos]
                zs = ph["deviation_z"].values

                # Color each bar by z-score direction and magnitude
                colors_p = []
                for z in zs:
                    if z > 2:
                        colors_p.append("#16a34a")   # dark green — significantly hot
                    elif z > 1:
                        colors_p.append("#86efac")   # light green — slightly hot
                    elif z < -2:
                        colors_p.append("#e05252")   # red — significantly cold
                    elif z < -1:
                        colors_p.append("#f59e0b")   # amber — slightly cold
                    else:
                        colors_p.append("#4C72B0")   # blue — normal

                expected_count = ph["count"].sum() / len(ph)
                fig_p = go.Figure(go.Bar(
                    x=ph["digit"].astype(str),
                    y=ph["count"],
                    marker_color=colors_p,
                    hovertemplate=(
                        "Digit: %{x}<br>"
                        "Count: %{y}<br>"
                        f"Expected: {expected_count:.0f}<br>"
                        "Z-score: %{customdata:.2f}<extra></extra>"
                    ),
                    customdata=zs,
                ))
                fig_p.add_hline(
                    y=expected_count,
                    line_dash="dash", line_color="#55A868",
                    annotation_text=f"Expected ({expected_count:.0f})",
                    annotation_position="top right",
                )
                fig_p.update_layout(
                    xaxis_title="Digit / Number",
                    yaxis_title="Times drawn in this position",
                    title=f"Position {pos} Frequency{' — ' + draw_time_label if draw_time_label else ''}",
                )
                st.plotly_chart(fig_p, use_container_width=True)

                # --- Plain-text interpretation ---
                top = ph.loc[ph["count"].idxmax()]
                bot = ph.loc[ph["count"].idxmin()]
                hot_nums = ph[ph["deviation_z"] > 2]["digit"].tolist()
                cold_nums = ph[ph["deviation_z"] < -2]["digit"].tolist()

                lines = [
                    f"**Position {pos} summary{' (' + draw_time_label + ')' if draw_time_label else ''}:**",
                    f"- Most frequent: digit **{int(top['digit'])}** "
                    f"({int(top['count'])} draws, {top['frequency_pct']:.1f}% — "
                    f"z={top['deviation_z']:+.2f})",
                    f"- Least frequent: digit **{int(bot['digit'])}** "
                    f"({int(bot['count'])} draws, {bot['frequency_pct']:.1f}% — "
                    f"z={bot['deviation_z']:+.2f})",
                ]
                if hot_nums:
                    lines.append(
                        f"- **Statistically above expected** (z > +2): digits {hot_nums} — "
                        "appeared more often than chance suggests in this position."
                    )
                if cold_nums:
                    lines.append(
                        f"- **Statistically below expected** (z < −2): digits {cold_nums} — "
                        "appeared less often than chance suggests in this position."
                    )
                if not hot_nums and not cold_nums:
                    lines.append(
                        "- All digits are within the normal statistical range for this position. "
                        "No significant bias detected."
                    )
                lines.append(
                    "_Note: These are historical counts only. Each draw is an independent random event — "
                    "past frequency does not predict future draws._"
                )
                st.info("\n".join(lines))

    with st.spinner("Computing heat map…"):
        if game_cfg.ordered:
            st.caption("Position-specific frequency analysis — each chart shows how often each digit appeared in that position across all draws.")
            if is_multi_draw:
                pos_by_time = {
                    dt: _pos_heat_map(
                        df[df["draw_time"] == dt].tail(window) if window else df,
                        game_cfg, draw_time=dt,
                    )
                    for dt in game_cfg.draw_times
                }
                tab_labels = [_TIME_LABELS.get(dt, dt) for dt in game_cfg.draw_times]
                tabs_dt = st.tabs(tab_labels)
                for tab, dt in zip(tabs_dt, game_cfg.draw_times):
                    with tab:
                        _draw_position_charts(pos_by_time[dt], _TIME_LABELS.get(dt, dt))
                heat = pd.DataFrame()
                heat_by_time = {dt: pd.DataFrame() for dt in game_cfg.draw_times}
            else:
                pos_heat_single = _pos_heat_map(df.tail(window) if window else df, game_cfg)
                _draw_position_charts(pos_heat_single)
                heat = pd.DataFrame()
                heat_by_time = {}
        elif is_multi_draw:
            heat_by_time = {
                dt: number_heat_map(df, game_cfg, window=window, draw_time=dt)
                for dt in game_cfg.draw_times
            }
            tab_labels = [_TIME_LABELS.get(dt, dt) for dt in game_cfg.draw_times]
            tabs = st.tabs(tab_labels)
            for tab, dt in zip(tabs, game_cfg.draw_times):
                with tab:
                    h = heat_by_time[dt]
                    _draw_heat_chart(h)
                    with st.expander("Full table"):
                        st.dataframe(h, use_container_width=True)
            heat = heat_by_time[game_cfg.draw_times[0]]
        else:
            heat = number_heat_map(df, game_cfg, window=window)
            heat_by_time = {}
            _draw_heat_chart(heat)
            if not heat.empty:
                with st.expander("Full heat map table"):
                    st.dataframe(heat, use_container_width=True)

    # --- Local combinations: learned scoring or explicit uniform sampling ---
    st.subheader("Combinations")
    generation_mode = st.selectbox(
        "Selection method", ["ML-assisted", "Uniform random"],
        index=0, key="prob_generation_mode",
    )
    st.caption(
        "ML-assisted selection uses historical logistic-regression scores as sampling weights. "
        "At least 100 valid draws per slot are required; 30/90-draw windows fall back to uniform random. "
        "Scores are not ticket odds and do not establish a winning advantage. **Not a prediction.**"
        if generation_mode == "ML-assisted" else
        "Uniform random sampling — no learned weights. Every valid combination has the same "
        "chance in a fair draw. **Not a prediction.**"
    )

    col_sl, col_lbl = st.columns([3, 5])
    with col_sl:
        n_combos = st.slider("Number of combinations", 1, 50, 5, key="n_combos")
    with col_lbl:
        st.caption(
            f"Each combination picks **{game_cfg.picks} numbers** "
            f"from {game_cfg.low_boundary}–{game_cfg.pool} ({game_cfg.name}). "
            + ("Position order is preserved; repeated digits are allowed."
               if game_cfg.ordered else "Numbers do not repeat within a combination.")
        )

    _combo_fmt = (
        (lambda combo: " – ".join(f"`{d:02d}`" for d in combo))
        if game_cfg.ordered else
        (lambda combo: " &nbsp;–&nbsp; ".join(f"`{n:02d}`" for n in combo))
    )

    _ai_combo_key = (
        f"ai_combo_lean_v1_{_LEAN_ML_VERSION}_{game_id}_{window}_{_data_hash}"
        f"_{generation_mode}_{n_combos}"
    )
    # One source of truth for display and the optional AI explanation.
    with st.spinner("Preparing combinations and chronological evaluation…"):
        generation_by_label = {
            _TIME_LABELS.get(dt, dt): _prob_generation(
                df, _data_hash, game_id, window, dt, n_combos,
                generation_mode, _LEAN_ML_VERSION,
            )
            for dt in game_cfg.draw_times
        }

    def _render_generation(result: dict) -> None:
        for i, combo in enumerate(result["combinations"], 1):
            st.markdown(f"**#{i}** &nbsp; " + _combo_fmt(combo))
        evaluation = result["evaluation"]
        if result["method"] == "uniform_fallback":
            st.warning(result["reason"])
        metrics = st.columns(3)
        for column, label, value in zip(
            metrics,
            ["Test model Brier", "Uniform Brier", "Test draws"],
            [evaluation.get("model_brier"), evaluation.get("baseline_brier"),
             evaluation.get("test_draws", 0)],
        ):
            column.metric(label, "N/A" if value is None else
                          (str(value) if label == "Test draws" else f"{value:.6f}"))
        interval = evaluation.get("improvement_ci95", [])
        st.caption(
            "Lower Brier loss is better. 95% CI for uniform minus model loss: "
            + (f"[{interval[0]:.6f}, {interval[1]:.6f}]." if len(interval) == 2
               else "not evaluated.")
            + " Positive values favor the model; this is not evidence of improved lottery odds."
        )
        if evaluation.get("evidence_of_improvement"):
            st.info("Historical test Brier improvement observed; no demonstrated winning edge.")
        else:
            st.info("No demonstrated edge over uniform sampling.")
        st.caption(
            f"Evaluation status: {evaluation['status']} · "
            f"Valid windowed draws: {evaluation.get('usable_draws', 0):,} · "
            f"Production training draws: {result['training_draws']:,} "
            f"(excludes the {_lean_ml.BURN_IN}-draw burn-in)."
        )
        with st.expander("Per-number model scores and feature weights"):
            if result["scores"]:
                st.dataframe(pd.DataFrame(result["scores"]), hide_index=True,
                             use_container_width=True)
                st.caption("Marginal scores used for sampling, not joint ticket probabilities.")
                st.dataframe(pd.DataFrame([
                    {"feature": name, "coefficient": weight}
                    for name, weight in result["feature_weights"].items()
                ]), hide_index=True, use_container_width=True)
                st.caption("Standardized logistic coefficients are associations, not causal effects.")
            else:
                st.caption("No learned scores or coefficients for uniform sampling.")
            if evaluation.get("caution"):
                st.caption(evaluation["caution"])

    if is_multi_draw:
        combo_tabs = st.tabs(list(generation_by_label))
        for tab, result in zip(combo_tabs, generation_by_label.values()):
            with tab:
                _render_generation(result)
    else:
        _render_generation(next(iter(generation_by_label.values())))

    # --- Azure AI Analysis ---
    st.subheader("AI Statistical Interpretation")
    st.caption(
        f"Analysis based on **{len(df):,} draws** up to **{_prob_latest_date}**. "
        "Result persists this session — click Regenerate or Update Data for a fresh analysis."
    )

    ai_client = AzureAIClient()
    if not ai_client.is_configured():
        st.warning(ai_client.setup_instructions())
    else:
        st.caption(
            f"Endpoint: `{ai_client._endpoint}`  |  "
            f"Deployment: **{ai_client._deployment}**  |  "
            f"API version: `{ai_client._api_version}`"
        )

        def _build_slots_data():
            """Use the generator's validated history and exact displayed tickets."""
            slots = []
            for dt in game_cfg.draw_times:
                label = _TIME_LABELS.get(dt, dt)
                clean = _lean_ml._prepare_draws(df, game_cfg, window, dt)
                cols = [f"n{i}" for i in range(1, game_cfg.picks + 1)]
                rows = []
                groups = (
                    [(pos, clean[col].value_counts()) for pos, col in enumerate(cols, 1)]
                    if game_cfg.ordered else
                    [(None, pd.Series(clean[cols].to_numpy().ravel()).value_counts())]
                )
                for position, counts in groups:
                    for number in range(game_cfg.low_boundary, game_cfg.pool + 1):
                        rows.append({
                            "number": number, "total_draws": int(counts.get(number, 0)),
                            **({"position": position} if position is not None else {}),
                        })
                slots.append({
                    "label": label, "heat_map_rows": rows, "total_draws": len(clean),
                    "generation": generation_by_label[label],
                })
            return slots

        def _render_ai_results(ai_results: dict) -> None:
            for slot_label, res in ai_results.items():
                if is_multi_draw:
                    st.markdown(f"#### {slot_label}")
                if res.get("combination"):
                    combo_str = " — ".join(f"**{n:02d}**" for n in res["combination"])
                    heading = (
                        "ML-assisted Combination — AI Explanation"
                        if res.get("generation", {}).get("method") == "logistic_regression"
                        else "Uniform random Combination — AI Explanation"
                    )
                    st.success(
                        f"### {heading}\n\n"
                        f"{combo_str}\n\n*{game_cfg.name}*"
                        + (f" · {slot_label}" if is_multi_draw else "")
                    )
                    # Save as Bet — unique key per slot
                    _save_key = f"save_ai_{game_cfg.id}_{slot_label}"
                    _label_key = f"save_label_{game_cfg.id}_{slot_label}"
                    _bet_date_key = f"save_betdate_{game_cfg.id}_{slot_label}"
                    import datetime as _dt_mod2

                    # --- Auto-generate label and next draw date ---
                    _WDAY = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3,
                             "Fri": 4, "Sat": 5, "Sun": 6}
                    _REVERSE_TL = {"2:00 PM": "14:00", "5:00 PM": "17:00", "9:00 PM": "21:00"}
                    _draw_days_idx = {_WDAY[d] for d in game_cfg.draw_days}
                    _today = _dt_mod2.date.today()
                    _next_draw = next(
                        (_today + _dt_mod2.timedelta(days=i)
                         for i in range(1, 8)
                         if (_today + _dt_mod2.timedelta(days=i)).weekday() in _draw_days_idx),
                        _today + _dt_mod2.timedelta(days=1),
                    )
                    _game_slug = game_cfg.name.replace(" ", "").replace("/", "")
                    if is_multi_draw:
                        _dt_str = _REVERSE_TL.get(slot_label, slot_label).replace(":", "")
                        _auto_label = f"{_next_draw.strftime('%Y%m%d')}_{_dt_str}_{_game_slug}"
                    else:
                        _auto_label = f"{_next_draw.strftime('%Y%m%d')}_{_game_slug}"

                    _sc1, _sc2 = st.columns([2, 1])
                    with _sc1:
                        _label_val = st.text_input(
                            "Note (optional)",
                            key=_label_key,
                            value=_auto_label,
                            label_visibility="collapsed",
                        )
                    with _sc2:
                        _bet_date_val = st.date_input(
                            "Draw date",
                            key=_bet_date_key,
                            value=_next_draw,
                            min_value=_today,
                            format="YYYY-MM-DD",
                            label_visibility="collapsed",
                        )
                    if st.button("🎯 Save as Bet", key=_save_key):
                        _combo_formatted = "-".join(f"{n:02d}" for n in res["combination"])
                        _dt_save = slot_label if is_multi_draw else None
                        try:
                            _conn_sv = get_connection()
                            _conn_sv.execute(
                                """INSERT INTO saved_combinations
                                   (id, saved_at, game, draw_time, combination, reasoning,
                                    label, source, bet_date, status, ml_run_id)
                                   VALUES (nextval('saved_combinations_seq'), NOW(),
                                           ?, ?, ?, ?, ?, 'bet', ?, 'pending', ?)""",
                                [game_cfg.id, _dt_save, _combo_formatted,
                                 res.get("reasoning"), _label_val or None,
                                 _bet_date_val, None],
                            )
                            _conn_sv.close()
                            st.toast(f"Bet saved: {_combo_formatted}", icon="🎯")
                        except Exception as _sv_exc:
                            st.error(f"Save failed: {_sv_exc}")
                if res.get("reasoning"):
                    st.info(res["reasoning"])
                if res.get("disclaimer"):
                    st.caption(f"⚠️ {res['disclaimer']}")

                if not res.get("combination") and not res.get("reasoning"):
                    st.warning("AI returned an empty response. Click Regenerate to try again.")

        generate_requested = st.button("Generate / Regenerate AI Explanation", key="regen_ai")
        if generate_requested:
            st.session_state.pop(_ai_combo_key, None)
            st.session_state.pop(f"{_ai_combo_key}_prompt", None)

        if _ai_combo_key in st.session_state:
            st.caption(f"Based on **{len(df):,} draws** up to **{_prob_latest_date}**.")
            _render_ai_results(st.session_state[_ai_combo_key])
        elif generate_requested:
            with st.spinner("AI is analysing latest draws — this may take a moment…"):
                try:
                    if game_cfg.ordered:
                        chi_result = _an.digit_uniformity(df, game_cfg).conclusion
                        gap_result = _an.digit_repeat_rate(df, game_cfg).conclusion
                    else:
                        chi_result = _an.freq_chi_square(df, game_cfg).conclusion
                        gap_result = _an.gap_analysis(df, game_cfg).conclusion

                    _slots = [slot for slot in _build_slots_data() if slot["heat_map_rows"]]
                    ai_results = ai_client.interpret_lotto_stats(
                        game_name=game_cfg.name,
                        slots=_slots,
                        chi_sq_result=chi_result,
                        gap_result=gap_result,
                        picks=game_cfg.picks,
                    )
                    st.session_state[_ai_combo_key] = ai_results
                    st.session_state[f"{_ai_combo_key}_prompt"] = ai_client.last_prompt
                    st.rerun()
                except Exception as exc:
                    st.error(f"Azure AI error: {exc}")

    # ---- Same-Date Historical Comparison — All Games ----
    import datetime as _dt_cmp
    _cmp_today = _dt_cmp.date.today()
    _cmp_month_name = _cmp_today.strftime("%B")
    _cmp_day_num = _cmp_today.day

    st.markdown("---")
    st.subheader(f"📅 On This Day — {_cmp_month_name} {_cmp_day_num}, All Years & {game_cfg.name}")
    st.caption(
        "Actual winning numbers drawn on this calendar date in every year since this game started. "
        + ("Numbers in the current AI combination are highlighted **orange**."
           if _ai_combo_key in st.session_state else "Run AI analysis above to highlight matching numbers.")
    )

    # Collect AI combination numbers from all draw-time slots (empty set if AI not run yet)
    _cmp_ai_nums: set[int] = set()
    for _cslot in st.session_state.get(_ai_combo_key, {}).values():
        _cmp_ai_nums.update(int(n) for n in _cslot.get("combination", []))

    # Build table for the selected game only
    _cmp_all_rows: list[dict] = []
    for _cgid, _cgcfg in [(game_cfg.id, game_cfg)]:
        try:
            _cgdf = load_draws(_cgid)
        except Exception:
            continue
        if _cgdf.empty:
            continue
        _cgdd = pd.to_datetime(_cgdf["draw_date"])
        _cg_same = _cgdf[
            (_cgdd.dt.month == _cmp_today.month) & (_cgdd.dt.day == _cmp_today.day)
        ].sort_values("draw_date")
        if _cg_same.empty:
            continue

        _cg_num_cols = (
            [f"pub_{i}" for i in range(1, _cgcfg.picks + 1)] if _cgcfg.ordered
            else [f"n{i}" for i in range(1, _cgcfg.picks + 1)]
        )
        _cg_num_cols = [c for c in _cg_num_cols if c in _cg_same.columns]

        for _, _crow in _cg_same.iterrows():
            _cnums = [int(_crow[c]) for c in _cg_num_cols if pd.notna(_crow[c])]
            _cmatches = len(set(_cnums) & _cmp_ai_nums) if _cmp_ai_nums else None
            _cmp_all_rows.append({
                "Game":      _cgcfg.name,
                "Year":      int(str(_crow["draw_date"])[:4]),
                "Draw Date": str(_crow["draw_date"])[:10],
                "Numbers":   "  ".join(f"{n:02d}" for n in _cnums),
                "Matches":   _cmatches,
            })

    if not _cmp_all_rows:
        st.info(
            f"No draws recorded on {_cmp_month_name} {_cmp_day_num} for any game yet. "
            "Try loading more historical data using the sidebar."
        )
    else:
        _cmp_tbl = pd.DataFrame(_cmp_all_rows)

        # Summary metrics
        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.metric("Total draws on this date", len(_cmp_tbl))
        _mc2.metric("Years with data", _cmp_tbl["Year"].nunique())
        if _cmp_ai_nums and _cmp_tbl["Matches"].notna().any():
            _mc3.metric(
                "Best match (this game)",
                f"{int(_cmp_tbl['Matches'].max(skipna=True))} / {game_cfg.picks}"
            )

        # Row styler
        def _cmp_style(row):
            m = row.get("Matches")
            if m is not None and not (isinstance(m, float) and pd.isna(m)) and int(m) >= 3:
                return ["background-color:#dcfce7; color:#14532d"] * len(row)
            if m is not None and not (isinstance(m, float) and pd.isna(m)) and int(m) >= 1:
                return ["background-color:#fef3c7; color:#92400e"] * len(row)
            return [""] * len(row)

        _display_tbl = _cmp_tbl.copy()
        if not _cmp_ai_nums:
            _display_tbl = _display_tbl.drop(columns=["Matches"])

        _cmp_col_cfg: dict = {
            "Game":      st.column_config.TextColumn("Game",      width="medium"),
            "Year":      st.column_config.NumberColumn("Year",    format="%d", width="small"),
            "Draw Date": st.column_config.TextColumn("Draw Date", width="medium"),
            "Numbers":   st.column_config.TextColumn("Numbers",   width="large"),
        }
        if _cmp_ai_nums:
            _cmp_col_cfg["Matches"] = st.column_config.NumberColumn("Matches", format="%d", width="small")

        st.dataframe(
            _display_tbl.style.apply(_cmp_style, axis=1),
            use_container_width=True,
            height=min(520, 38 + 35 * len(_display_tbl)),
            column_config=_cmp_col_cfg,
            hide_index=True,
        )

        # Frequency chart for selected game
        with st.expander(f"Number frequency breakdown — {game_cfg.name}"):
            for _fgame_name in _cmp_tbl["Game"].unique():
                _frows = _cmp_tbl[_cmp_tbl["Game"] == _fgame_name]
                _fnums_flat: list[int] = []
                for _nr in _frows["Numbers"]:
                    _fnums_flat.extend(int(x) for x in str(_nr).split() if x.isdigit())
                if not _fnums_flat:
                    continue
                _ffreq = (
                    pd.Series(_fnums_flat)
                    .value_counts()
                    .sort_index()
                    .reset_index()
                )
                _ffreq.columns = ["number", "count"]
                _ffreq["type"] = _ffreq["number"].apply(
                    lambda n: "AI Pick" if n in _cmp_ai_nums else "Historical"
                )
                _ffig = px.bar(
                    _ffreq, x="number", y="count", color="type",
                    color_discrete_map={"AI Pick": "#f97316", "Historical": "#3b82f6"},
                    title=f"{_fgame_name} — {_cmp_month_name} {_cmp_day_num} (all years)",
                    labels={"number": "Number", "count": "Times drawn", "type": ""},
                    height=260,
                )
                _ffig.update_layout(
                    legend=dict(orientation="h", y=1.15),
                    margin=dict(t=50, b=20),
                )
                st.plotly_chart(_ffig, use_container_width=True)

    # ---- AI Combination vs All Historical Draws ----
    if _cmp_ai_nums:
        st.markdown("---")
        st.subheader(f"🔍 AI Combination vs All Historical Draws — {game_cfg.name}")
        st.caption(
            "How often the current AI-suggested numbers would have matched past draws "
            "across the entire recorded history of this game."
        )

        _hist_df = load_draws(game_cfg.id)
        if not _hist_df.empty:
            _h_num_cols = (
                [f"pub_{i}" for i in range(1, game_cfg.picks + 1)] if game_cfg.ordered
                else [f"n{i}" for i in range(1, game_cfg.picks + 1)]
            )
            _h_num_cols = [c for c in _h_num_cols if c in _hist_df.columns]

            _hist_rows: list[dict] = []
            for _, _hr in _hist_df.iterrows():
                _hnums = [int(_hr[c]) for c in _h_num_cols if pd.notna(_hr[c])]
                if not _hnums:
                    continue
                _hmatch = len(set(_hnums) & _cmp_ai_nums)
                _hist_rows.append({
                    "draw_date": str(_hr["draw_date"])[:10],
                    "Numbers":   "  ".join(f"{n:02d}" for n in _hnums),
                    "Matches":   _hmatch,
                })

            _hist_match_df = pd.DataFrame(_hist_rows)
            _total_draws = len(_hist_match_df)

            # Summary metrics per match count
            _match_summary = (
                _hist_match_df["Matches"]
                .value_counts()
                .sort_index(ascending=False)
                .reset_index()
            )
            _match_summary.columns = ["Matches", "Count"]
            _match_summary["% of Draws"] = (_match_summary["Count"] / _total_draws * 100).round(2)

            _hc1, _hc2, _hc3, _hc4 = st.columns(4)
            for _col, _min_match, _label in [
                (_hc1, game_cfg.picks, "Jackpot match"),
                (_hc2, game_cfg.picks - 1, f"{game_cfg.picks-1}-number match"),
                (_hc3, game_cfg.picks - 2, f"{game_cfg.picks-2}-number match"),
                (_hc4, 3, "3-number match"),
            ]:
                _cnt = int(_hist_match_df[_hist_match_df["Matches"] == _min_match]["Matches"].count())
                _col.metric(_label, _cnt, f"{_cnt/_total_draws*100:.2f}% of {_total_draws:,} draws")

            with st.expander("Top matching draws (3 or more matches)"):
                _top_matches = (
                    _hist_match_df[_hist_match_df["Matches"] >= 3]
                    .sort_values(["Matches", "draw_date"], ascending=[False, False])
                    .reset_index(drop=True)
                )
                if _top_matches.empty:
                    st.info("No draws with 3 or more matches found in the full history.")
                else:
                    def _hist_style(row):
                        m = int(row["Matches"])
                        if m == game_cfg.picks:
                            return ["background-color:#dcfce7; color:#14532d"] * len(row)
                        if m >= game_cfg.picks - 1:
                            return ["background-color:#dbeafe; color:#1e3a8a"] * len(row)
                        if m >= 3:
                            return ["background-color:#fef3c7; color:#92400e"] * len(row)
                        return [""] * len(row)
                    st.dataframe(
                        _top_matches.style.apply(_hist_style, axis=1),
                        use_container_width=True,
                        height=min(520, 38 + 35 * len(_top_matches)),
                        column_config={
                            "draw_date": st.column_config.TextColumn("Draw Date", width="medium"),
                            "Numbers":   st.column_config.TextColumn("Numbers",   width="large"),
                            "Matches":   st.column_config.NumberColumn("Matches", format="%d", width="small"),
                        },
                        hide_index=True,
                    )

            with st.expander("Match count distribution"):
                _mfig = px.bar(
                    _match_summary,
                    x="Matches", y="Count",
                    text="% of Draws",
                    labels={"Matches": "Numbers matched", "Count": "Draw count"},
                    title=f"AI combination match frequency — {game_cfg.name} ({_total_draws:,} draws)",
                    height=300,
                )
                _mfig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                _mfig.update_layout(margin=dict(t=50, b=20))
                st.plotly_chart(_mfig, use_container_width=True)

    # --- Full AI Prompt (copy & paste into other AI models) ---
    _saved_prompt = st.session_state.get(f"{_ai_combo_key}_prompt", "")
    if _saved_prompt:
        st.markdown("---")
        with st.expander("📋 Full prompt sent to AI — copy to run in Claude / Gemini / other models"):
            st.caption(
                "This is the exact prompt sent to the Azure OpenAI model. "
                "Paste it into Claude, Gemini, GPT-4, or any other AI to compare results."
            )
            st.code(_saved_prompt, language="text")


# ===========================================================================
# Page 6: Backtest
# ===========================================================================
elif page == "Backtest":
    import plotly.express as _px
    from src.lotto.backtest import run_backtest

    st.title("🔁 Strategy Backtest")
    st.caption(
        "Simulates the composite-score strategy against every historical draw. "
        "A net loss is expected — lotto draws are independent random events."
    )

    _b_game = game_selector("Game")
    _b_gcfg = GAMES[_b_game]
    _b_df = load_draws(_b_game)
    _b_df = _b_df.sort_values("draw_date")

    _b_col1, _b_col2, _b_col3 = st.columns(3)
    with _b_col1:
        _b_window = st.selectbox("Heat-map window", [None, 30, 60, 90, 180],
                                  format_func=lambda x: "All draws" if x is None else f"Last {x}")
    with _b_col2:
        _b_dt = None
        if len(_b_gcfg.draw_times) > 1:
            _b_dt = st.selectbox("Draw time", list(_b_gcfg.draw_times))
    with _b_col3:
        _b_maxrows = st.slider("Result rows shown", 50, 500, 200, 50)

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner("Running backtest — this may take a moment…"):
            _b_res = run_backtest(_b_df, _b_gcfg, window=_b_window,
                                   draw_time=_b_dt, max_rows=_b_maxrows)
        if "error" in _b_res:
            st.error(_b_res["error"])
        else:
            _b_summ = _b_res["summary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Draws tested", f"{_b_summ['total_draws_tested']:,}")
            c2.metric("Spent (₱)", f"{_b_summ['total_spent_php']:,.0f}")
            c3.metric("Won (₱)", f"{_b_summ['total_won_php']:,.0f}")
            roi_color = "normal" if _b_summ["roi_pct"] >= 0 else "inverse"
            c4.metric("ROI", f"{_b_summ['roi_pct']:.2f}%",
                       delta=f"₱{_b_summ['net_php']:,.0f}")

            st.markdown(f"**Best match:** {_b_summ['best_match']} numbers | "
                         f"**Avg matches/draw:** {_b_summ['avg_matches_per_draw']:.2f} | "
                         f"**Exact jackpot wins:** {_b_summ['exact_matches']}")

            _b_dist = _b_summ["match_distribution"]
            if _b_dist:
                _b_dist_df = pd.DataFrame(
                    {"matches": list(_b_dist.keys()), "draws": list(_b_dist.values())}
                ).sort_values("matches")
                st.plotly_chart(
                    _px.bar(_b_dist_df, x="matches", y="draws",
                            title="Match Distribution", labels={"matches": "Correct numbers matched",
                                                                  "draws": "# of draws"}),
                    use_container_width=True,
                )

            _b_rows = _b_res["results"]
            if _b_rows:
                _b_tbl = pd.DataFrame(_b_rows)
                _b_tbl["predicted"] = _b_tbl["predicted"].apply(lambda x: "-".join(map(str, x)))
                _b_tbl["actual"] = _b_tbl["actual"].apply(lambda x: "-".join(map(str, x)))
                st.dataframe(_b_tbl, use_container_width=True)


# ===========================================================================
# Page 7: Advanced Stats
# ===========================================================================
elif page == "Advanced Stats":
    import plotly.express as _px
    from src.lotto.analysis import (
        structural_break, entropy_score, power_analysis,
        cross_game_correlation, draw_fingerprint_pca,
    )

    st.title("📊 Advanced Statistical Analysis")
    _as_game = game_selector("Game")
    _as_gcfg = GAMES[_as_game]
    _as_df = load_draws(_as_game)

    _as_test = st.radio(
        "Analysis",
        ["Structural Break", "Entropy / Randomness", "Power Analysis",
         "Cross-Game Correlation", "PCA Fingerprint"],
        horizontal=True,
    )

    if _as_test == "Structural Break":
        _r = structural_break(_as_df, _as_gcfg)
        st.subheader(_r.name)
        st.caption(_r.conclusion)
        _t = _r.table.copy()
        fig = _px.bar(
            _t.melt(id_vars="number", value_vars=["freq_first_half", "freq_second_half"],
                    var_name="half", value_name="frequency"),
            x="number", y="frequency", color="half", barmode="group",
            title=f"First-half vs Second-half frequencies (split at {_r.meta['mid_date']})",
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Full table"):
            st.dataframe(_t, use_container_width=True)

    elif _as_test == "Entropy / Randomness":
        _r = entropy_score(_as_df, _as_gcfg)
        st.subheader(_r.name)
        st.caption(_r.conclusion)
        _m = _r.meta
        col1, col2 = st.columns(2)
        col1.metric("Observed entropy (bits)", f"{_m['obs_entropy']:.4f}")
        col2.metric("% of maximum", f"{_m['entropy_pct']:.1f}%")
        fig = _px.bar(_r.table, x="year", y="pct_of_max",
                      title="Entropy % of maximum by year",
                      labels={"pct_of_max": "% of max entropy", "year": "Year"})
        fig.add_hline(y=100, line_dash="dash", line_color="green",
                       annotation_text="Max (perfect randomness)")
        st.plotly_chart(fig, use_container_width=True)

    elif _as_test == "Power Analysis":
        _r = power_analysis(_as_df, _as_gcfg)
        st.subheader(_r.name)
        st.caption(_r.conclusion)
        _t = _r.table.copy()
        fig = _px.scatter(
            _t, x="number", y="observed",
            error_y=_t["ci_95_hi"] - _t["observed"],
            error_y_minus=_t["observed"] - _t["ci_95_lo"],
            title="Observed frequency with 95% Wilson CI",
        )
        fig.add_hline(y=_t["expected"].iloc[0] if not _t.empty else 0,
                       line_dash="dash", line_color="orange",
                       annotation_text="Expected")
        st.plotly_chart(fig, use_container_width=True)
        _n_out = _r.meta["n_outside_ci"]
        if _n_out > 0:
            st.warning(f"{_n_out} numbers outside their 95% CI — may indicate non-uniformity.")
        with st.expander("Full table"):
            st.dataframe(_t, use_container_width=True)

    elif _as_test == "Cross-Game Correlation":
        st.info("Loading all games — this may take a few seconds…")
        _all_dfs = {gid: load_draws(gid) for gid in GAMES}
        _corr = cross_game_correlation(_all_dfs, GAMES)
        if _corr.empty:
            st.warning("Not enough shared draw dates to compute correlations.")
        else:
            st.subheader("Spearman correlation of daily draw sums across games")
            fig = _px.imshow(
                _corr, text_auto=True, color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, title="Cross-game correlation (Spearman ρ)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Values near 0 = independent. High positive correlation may indicate "
                "draws happened on the same dates rather than a true statistical link."
            )

    elif _as_test == "PCA Fingerprint":
        with st.spinner("Running PCA…"):
            _pca = draw_fingerprint_pca(_as_df, _as_gcfg)
        if "error" in _pca:
            st.error(_pca["error"])
        else:
            _ev = _pca["explained_variance"]
            st.caption(
                f"PC1 explains {_ev[0]*100:.1f}% variance, "
                f"PC2 explains {_ev[1]*100:.1f}% variance (total {sum(_ev)*100:.1f}%)"
            )
            _coords = _pca["coords"]
            fig = _px.scatter(
                _coords, x="pc1", y="pc2", color="year",
                hover_data=["draw_date", "draw_sum"],
                title="Draw fingerprint (PCA 2D projection)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each dot is one draw projected onto two principal components. "
                "Clusters may reflect seasonal or structural patterns, but do NOT imply predictability."
            )


# ===========================================================================
# Page 8: Combo Tools
# ===========================================================================
elif page == "Combo Tools":
    import plotly.express as _px
    from src.lotto.predict import combination_quality, score_user_combination, suggest_combinations
    from src.lotto.wheel import (
        full_wheel, full_wheel_cost, abbreviated_wheel, syndicate_tickets, wheel_stats
    )

    st.title("🎯 Combination Tools")
    _ct_game = game_selector("Game")
    _ct_gcfg = GAMES[_ct_game]
    _ct_df = load_draws(_ct_game)

    _ct_tab1, _ct_tab2, _ct_tab3, _ct_tab4 = st.tabs(
        ["Your Numbers Analyzer", "Wheeling System", "Syndicate Optimizer",
         "🔎 Combination History"]
    )

    with _ct_tab1:
        st.subheader("Your Numbers Analyzer")
        st.caption("Enter your chosen numbers and see how they score statistically.")
        _pool = list(range(_ct_gcfg.low_boundary, _ct_gcfg.pool + 1))
        _user_nums = st.multiselect(
            f"Select exactly {_ct_gcfg.picks} numbers",
            _pool, max_selections=_ct_gcfg.picks,
        )
        if len(_user_nums) == _ct_gcfg.picks:
            _score = score_user_combination(_user_nums, _ct_gcfg, _ct_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Quality score", f"{_score['quality_score']}/100")
            c2.metric("Verdict", _score["verdict"])
            c3.metric("Sum percentile", f"{_score['sum_percentile']}%")
            c4.metric("Odd / Even", f"{_score['odd_count']} / {_score['even_count']}")
            for line in _score.get("commentary", []):
                st.info(line)
        elif _user_nums:
            st.warning(f"Select {_ct_gcfg.picks - len(_user_nums)} more number(s).")

    with _ct_tab2:
        st.subheader("Wheeling System")
        _wh_keys = st.multiselect(
            "Select key numbers (min = picks, max 15)",
            list(range(_ct_gcfg.low_boundary, _ct_gcfg.pool + 1)),
            max_selections=15,
        )
        _wh_type = st.radio("Wheel type", ["Full wheel", "Abbreviated wheel"], horizontal=True)
        _wh_guarantee = 4
        if _wh_type == "Abbreviated wheel":
            _wh_guarantee = st.slider("Guarantee match level", 2, _ct_gcfg.picks - 1, 4)

        if len(_wh_keys) >= _ct_gcfg.picks:
            _cost_info = full_wheel_cost(len(_wh_keys), _ct_gcfg)
            st.info(
                f"Full wheel: **{_cost_info['n_tickets']:,}** tickets "
                f"(₱{_cost_info['cost_php']:,.0f})"
            )
            if st.button("Generate Tickets"):
                with st.spinner("Generating…"):
                    if _wh_type == "Full wheel":
                        _tickets = full_wheel(_wh_keys, _ct_gcfg)
                    else:
                        _tickets = abbreviated_wheel(_wh_keys, _ct_gcfg, _wh_guarantee)
                _stats = wheel_stats(_tickets, _ct_gcfg)
                c1, c2, c3 = st.columns(3)
                c1.metric("Tickets", _stats["n_tickets"])
                c2.metric("Cost (₱)", f"{_stats['cost_php']:,.0f}")
                c3.metric("Pool coverage", f"{_stats['pool_coverage_pct']}%")
                st.metric("Pair coverage", f"{_stats['pair_coverage_pct']}%")
                _tdf = pd.DataFrame(
                    ["-".join(map(str, t)) for t in _tickets], columns=["combination"]
                )
                st.dataframe(_tdf, use_container_width=True)
        else:
            st.warning(f"Select at least {_ct_gcfg.picks} key numbers.")

    with _ct_tab3:
        st.subheader("Syndicate Optimizer")
        st.caption("Generate diverse tickets for a group — minimize overlap to cover more of the pool.")
        _syn_n = st.slider("Number of tickets", 2, 50, 10)
        _syn_window = st.selectbox("Heat-map window", [None, 30, 60, 90],
                                    format_func=lambda x: "All draws" if x is None else f"Last {x}")
        if st.button("Generate Syndicate Tickets"):
            with st.spinner("Optimizing…"):
                _syn_tickets = syndicate_tickets(_ct_gcfg, _syn_n, _ct_df, window=_syn_window)
            _syn_stats = wheel_stats(_syn_tickets, _ct_gcfg)
            c1, c2, c3 = st.columns(3)
            c1.metric("Tickets", _syn_stats["n_tickets"])
            c2.metric("Cost (₱)", f"{_syn_stats['cost_php']:,.0f}")
            c3.metric("Pool coverage", f"{_syn_stats['pool_coverage_pct']}%")
            c1b, c2b = st.columns(2)
            c1b.metric("Pair coverage", f"{_syn_stats['pair_coverage_pct']}%")
            c2b.metric("Avg overlap", _syn_stats["avg_inter_ticket_overlap"])
            _syn_tdf = pd.DataFrame(
                ["-".join(map(str, t)) for t in _syn_tickets], columns=["combination"]
            )
            st.dataframe(_syn_tdf, use_container_width=True)

    with _ct_tab4:
        from src.lotto.predict import find_combination_history
        st.subheader("🔎 Combination History")
        st.caption(
            "Check whether a combination has ever been drawn before. "
            "Enter your numbers below and see every past draw where this exact combination won."
        )

        _ch_pool = list(range(_ct_gcfg.low_boundary, _ct_gcfg.pool + 1))
        _ch_nums = st.multiselect(
            f"Select exactly {_ct_gcfg.picks} numbers to look up",
            _ch_pool,
            max_selections=_ct_gcfg.picks,
            key="combo_history_nums",
        )

        if len(_ch_nums) == _ct_gcfg.picks:
            _ch_res = find_combination_history(_ch_nums, _ct_gcfg, _ct_df)
            _combo_str = " - ".join(str(n) for n in _ch_res["combo"])

            if _ch_res["found"]:
                st.error(
                    f"⚠️ **{_combo_str}** has been drawn **{_ch_res['times_drawn']} time(s)** before!",
                    icon="🚨",
                )
                c1, c2 = st.columns(2)
                c1.metric("First drawn", _ch_res["first_drawn"])
                c2.metric("Most recently", _ch_res["last_drawn"])

                st.markdown("#### All matching draws")
                _ch_df = _ch_res["draws"].copy()
                # Rename for display
                _ch_df.columns = [c.replace("_", " ").title() for c in _ch_df.columns]
                st.dataframe(_ch_df, use_container_width=True, hide_index=True)

                st.info(
                    "This combination has appeared before, but that does **not** make it "
                    "more or less likely in future draws. Each draw is an independent "
                    "random event with no memory of past results."
                )
            else:
                st.success(
                    f"✅ **{_combo_str}** has **never been drawn** in this game's recorded history.",
                    icon="🎯",
                )
                st.info(
                    "This combination has no prior win on record, but that does **not** make "
                    "it more likely to be drawn. All combinations have equal probability."
                )
        elif _ch_nums:
            st.warning(f"Select {_ct_gcfg.picks - len(_ch_nums)} more number(s) to search.")


# ===========================================================================
# Page 9: Gambler's Edge
# ===========================================================================
elif page == "Gambler's Edge":
    import plotly.express as _px
    from src.lotto.ev import expected_value, multi_game_ev, jackpots_from_db, MIN_JACKPOTS
    from src.lotto.analysis import jackpot_rollover_prob, never_drawn_patterns

    st.title("🎰 Gambler's Edge")
    st.caption(
        "Mathematical tools for smarter play. All EV values are **negative** — "
        "this is expected for any lottery. The goal is to understand the math."
    )

    _ge_tab1, _ge_tab2, _ge_tab3, _ge_tab4, _ge_tab5 = st.tabs(
        ["EV Calculator", "Jackpot Rollover", "Never-Drawn Patterns",
            "Bias Detector", "Manual Wheel"]
    )

    with _ge_tab1:
        st.subheader("Expected Value Calculator")
        _ev_game = game_selector("Game")
        _ev_gcfg = GAMES[_ev_game]
        _ev_min = MIN_JACKPOTS.get(_ev_game, 0)
        # Auto-load latest jackpot from DB; fall back to minimum if not available
        try:
            _live_jps = jackpots_from_db(get_connection())
            _ev_live = _live_jps.get(_ev_game)
        except Exception:
            _ev_live = None
        _ev_default = float(_ev_live if _ev_live else _ev_min)
        _ev_jp = st.number_input(
            "Current jackpot (₱)",
            min_value=0.0,
            value=_ev_default,
            step=1_000_000.0,
            format="%.0f",
            help="Auto-loaded from latest draw in DB. Edit to adjust.",
        )
        if _ev_live:
            st.caption(f"📡 Live jackpot from DB: ₱{_ev_live:,.0f}")

        _ev_res = expected_value(_ev_gcfg, jackpot_php=_ev_jp)
        c1, c2, c3 = st.columns(3)
        c1.metric("EV per ticket (₱)", f"{_ev_res['total_ev_php']:.4f}")
        c2.metric("Net EV (₱)", f"{_ev_res['net_ev_php']:.4f}")
        c3.metric("Return %", f"{_ev_res['return_pct']:.2f}%")
        st.metric("Breakeven jackpot (₱)",
                   f"₱{_ev_res['breakeven_jackpot_php']:,.0f}")

        _tiers_df = pd.DataFrame(_ev_res["tiers"])
        if not _tiers_df.empty:
            fig = _px.bar(
                _tiers_df, x="matches", y="ev_contribution_php",
                text="odds_str",
                title="EV contribution by prize tier",
                labels={"matches": "Numbers matched", "ev_contribution_php": "EV contribution (₱)"},
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(_tiers_df[["matches", "odds_str", "prize_php", "ev_contribution_php"]],
                      use_container_width=True)

        st.subheader("Multi-Game EV Comparison")
        _mv_res = multi_game_ev()
        _mv_df = pd.DataFrame(
            [{"game": r["game_name"], "net_ev_php": round(r["net_ev_php"], 4),
               "return_pct": round(r["return_pct"], 2),
               "breakeven_jp": f"₱{r['breakeven_jackpot_php']:,.0f}"}
             for r in _mv_res]
        )
        st.dataframe(_mv_df, use_container_width=True)
        st.caption("Sorted by net EV descending (least bad first) at minimum jackpot.")

    with _ge_tab2:
        st.subheader("Jackpot Rollover Probability")
        _jr_game = game_selector("Game (rollover)")
        _jr_gcfg = GAMES[_jr_game]
        _jr_df = load_draws(_jr_game)
        _jr_res = jackpot_rollover_prob(_jr_df, _jr_gcfg)
        st.caption(_jr_res.conclusion)
        fig = _px.line(
            _jr_res.table, x="consecutive_rollovers", y="probability_pct",
            title="Probability of N consecutive rollovers (%)",
            labels={"consecutive_rollovers": "Consecutive rollovers", "probability_pct": "Probability (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Probability table"):
            st.dataframe(_jr_res.table, use_container_width=True)

    with _ge_tab3:
        st.subheader("Never-Drawn Patterns")
        _nd_game = game_selector("Game (patterns)")
        _nd_gcfg = GAMES[_nd_game]
        _nd_df = load_draws(_nd_game)
        with st.spinner("Analyzing…"):
            _nd_res = never_drawn_patterns(_nd_df, _nd_gcfg)
        st.caption(_nd_res.conclusion)
        _nd_meta = _nd_res.meta
        col1, col2 = st.columns(2)
        if "never_count" in _nd_meta:
            col1.metric("Never drawn", f"{_nd_meta['never_count']:,}")
            col2.metric("Total possible", f"{_nd_meta['total_possible']:,}")
        elif "never_pairs" in _nd_meta:
            col1.metric("Never-drawn pairs", f"{_nd_meta['never_pairs']:,}")
            col2.metric("Total pairs", f"{_nd_meta['total_pairs']:,}")
        if not _nd_res.table.empty:
            st.dataframe(_nd_res.table.head(200), use_container_width=True)

    # ---- Tab 4: Sequential Bias Detector ----
    with _ge_tab4:
        from src.lotto.analysis import sequential_bias_test as _seq_bias
        st.subheader("📈 Sequential Bias Detector")
        st.caption(
            "Tracks each number's cumulative frequency vs statistically expected over every draw. "
            "A fair draw keeps all Z-scores within **±2σ** ~95% of the time. "
            "Sustained deviation is a mechanical bias signal — the longer it persists, the less likely it is noise. "
            "**This is a fairness audit, not a prediction tool.**"
        )
        _sb_game = game_selector("Game (bias)")
        _sb_gcfg = GAMES[_sb_game]
        _sb_df = load_draws(_sb_game)
        with st.spinner("Running sequential bias test…"):
            _sb_res = _seq_bias(_sb_df, _sb_gcfg)

        _sb_meta = _sb_res.meta
        _sb_c1, _sb_c2, _sb_c3 = st.columns(3)
        _sb_c1.metric("Draws analysed", f"{_sb_meta['n_draws']:,}")
        _sb_c2.metric("Hot numbers (Z>2)", len(_sb_meta["hot_numbers"]))
        _sb_c3.metric("Cold numbers (Z<-2)", len(_sb_meta["cold_numbers"]))
        st.info(_sb_res.conclusion)

        # Z-score bar chart
        import plotly.express as _px_ge
        _sb_fig = _px_ge.bar(
            _sb_res.table,
            x="number", y="z_score",
            color="bias",
            color_discrete_map={"HOT 🔴": "#ef4444", "COLD 🔵": "#3b82f6", "normal ✅": "#6b7280"},
            title=f"Z-score per number — {_sb_gcfg.name} ({_sb_meta['n_draws']:,} draws)",
            labels={"number": "Number", "z_score": "Z-score"},
        )
        _sb_fig.add_hline(y=2, line_dash="dash", line_color="#ef4444", annotation_text="+2σ")
        _sb_fig.add_hline(y=-2, line_dash="dash", line_color="#3b82f6", annotation_text="-2σ")
        _sb_fig.update_layout(height=400)
        st.plotly_chart(_sb_fig, use_container_width=True)

        # Top biased numbers
        _sb_top = _sb_res.table[_sb_res.table["significance"] != "ns"]
        if not _sb_top.empty:
            st.markdown("**Statistically significant deviations (p<0.05)**")
            st.dataframe(_sb_top[["number", "appearances", "expected", "z_score", "bias", "significance"]],
                         hide_index=True, use_container_width=True)
        else:
            st.success("No numbers show statistically significant deviation from expected frequency.")

        with st.expander("Full table (all numbers)"):
            st.dataframe(_sb_res.table, hide_index=True, use_container_width=True)

    # ---- Tab 5: Manual Wheel ----
    with _ge_tab5:
        from src.lotto.wheel import full_wheel, abbreviated_wheel, wheel_stats as _wstats
        from src.lotto.ev import TICKET_PRICES
        from math import comb as _mcomb

        st.subheader("⚙️ Manual Wheel")
        st.caption(
            "Choose your own key numbers for a full or abbreviated wheel. Coverage is "
            "conditional on winning numbers falling within your key set: a full wheel "
            "contains the jackpot combination only if all six winning numbers are in the set; "
            "an abbreviated wheel guarantees at least a 4-number match only if at least "
            "four winning numbers are in the set. More tickets cost more. "
            "Wheeling offers no predictive advantage and does not improve any individual ticket's odds."
        )
        _mw_game = st.selectbox(
            "Game (wheel)",
            [gid for gid, cfg in GAMES.items() if not cfg.ordered and cfg.picks == 6],
            format_func=lambda gid: GAMES[gid].name,
            key="mw_game",
        )
        _mw_gcfg = GAMES[_mw_game]

        _mw_c1, _mw_c2 = st.columns(2)
        with _mw_c1:
            _mw_keys = st.multiselect(
                "Select key numbers (6–15)",
                list(range(_mw_gcfg.low_boundary, _mw_gcfg.pool + 1)),
                max_selections=15,
                key=f"mw_keys_{_mw_game}",
                help="Limited to 15 keys to keep wheel generation manageable.",
            )
        with _mw_c2:
            _mw_mode = st.radio("Wheel type", ["Full wheel", "Abbreviated (4-match guarantee)"], key="mw_mode")

        _mw_n_keys = len(_mw_keys)
        _mw_full_n = _mcomb(_mw_n_keys, _mw_gcfg.picks)
        _mw_over_limit = _mw_mode == "Full wheel" and _mw_full_n > 5000
        if _mw_n_keys < _mw_gcfg.picks:
            st.info(f"Select at least {_mw_gcfg.picks} key numbers to generate a wheel.")
        else:
            _mw_price = TICKET_PRICES.get(_mw_game, 20)
            st.caption(
                f"Full wheel = **{_mw_full_n:,} tickets** × ₱{_mw_price} = "
                f"**₱{_mw_full_n * _mw_price:,.0f}**. "
                "Abbreviated mode uses a reduced ticket set; its exact count and cost "
                "are shown after generation."
            )
        if _mw_over_limit:
            st.warning(
                f"Full wheel has {_mw_full_n:,} tickets — exceeds the 5,000-ticket limit. "
                "Reduce key numbers or use Abbreviated mode."
            )

        if st.button(
            "Generate Wheel Tickets", key="mw_generate",
            disabled=_mw_n_keys < _mw_gcfg.picks or _mw_over_limit,
        ):
            with st.spinner("Generating wheel…"):
                if _mw_mode == "Full wheel":
                    _mw_tickets = full_wheel(_mw_keys, _mw_gcfg)
                else:
                    _mw_tickets = abbreviated_wheel(_mw_keys, _mw_gcfg, guarantee=4)
                _mw_stats = _wstats(_mw_tickets, _mw_gcfg)

            _ws_c1, _ws_c2, _ws_c3 = st.columns(3)
            _ws_c1.metric("Tickets", f"{_mw_stats['n_tickets']:,}")
            _ws_c2.metric("Cost", f"₱{_mw_stats['cost_php']:,.0f}")
            _ws_c3.metric("Pool coverage", f"{_mw_stats['pool_coverage_pct']}%")
            st.caption("Pool coverage is the share of distinct numbers covered, not the chance of winning.")

            _ws_tdf = pd.DataFrame(
                [" – ".join(f"{n:02d}" for n in t) for t in _mw_tickets],
                columns=["combination"]
            )
            st.dataframe(_ws_tdf, use_container_width=True, hide_index=True)


# ===========================================================================
# Page 10: Data Quality
# ===========================================================================
elif page == "Data Quality":
    import plotly.express as _px
    from src.lotto.analysis import anomaly_detection

    st.title("🔍 Data Quality & Export")
    _dq_game = game_selector("Game")
    _dq_gcfg = GAMES[_dq_game]
    _dq_df = load_draws(_dq_game)

    st.subheader("Anomaly Detection")
    with st.spinner("Scanning for anomalies…"):
        _dq_res = anomaly_detection(_dq_df, _dq_gcfg)
    st.caption(_dq_res.conclusion)
    _dq_meta = _dq_res.meta
    c1, c2 = st.columns(2)
    c1.metric("Total draws", f"{_dq_meta['n_draws']:,}")
    c2.metric("Anomalies flagged", _dq_meta["n_anomalies"])
    if _dq_meta["n_anomalies"] > 0:
        st.warning("Anomalies detected — review the table below.")
        st.dataframe(_dq_res.table, use_container_width=True)
    else:
        st.success("No anomalies detected — all draws pass validation checks.")

    st.markdown("---")
    st.subheader("Export Data")
    _ex_col1, _ex_col2 = st.columns(2)
    with _ex_col1:
        _csv = _dq_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            data=_csv,
            file_name=f"pcso_{_dq_game}_draws.csv",
            mime="text/csv",
        )
    with _ex_col2:
        try:
            import io as _io
            _xls_buf = _io.BytesIO()
            _dq_df.to_excel(_xls_buf, index=False, engine="openpyxl")
            st.download_button(
                "⬇ Download Excel",
                data=_xls_buf.getvalue(),
                file_name=f"pcso_{_dq_game}_draws.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.info("Install openpyxl for Excel export: `pip install openpyxl`")

# ===========================================================================
elif page == "Saved Combos":
    import datetime as _dt_sv

    st.title("🎯 My Bets")
    st.caption(
        "AI-generated combinations you saved as bets from the **Probability & AI** page. "
        "Track status, check results against actual draws, and manage your bet history."
    )

    # Auto-refresh on every page load — silently match pending bets against actual draws
    try:
        _auto_updated = compare_with_actual()
        if _auto_updated:
            st.toast(f"Auto-updated {_auto_updated} bet(s) with actual draw results.", icon="✅")
    except Exception:
        pass

    _rf_col, _ = st.columns([1, 4])
    with _rf_col:
        if st.button("🔄 Refresh Results", key="refresh_actuals",
                     help="Manually re-compare all pending bets whose draw date has passed"):
            try:
                _n_updated = compare_with_actual()
                if _n_updated:
                    st.success(f"Updated {_n_updated} bet(s) with actual draw results.")
                else:
                    st.info("No new results found. Draws may not yet be loaded for those dates.")
            except Exception as _rf_exc:
                st.error(f"Refresh failed: {_rf_exc}")
            st.rerun()

    _STATUS_EMOJI = {"pending": "⏳", "won": "🏆", "lost": "❌", "cancelled": "🚫"}
    _STATUS_OPTIONS = ["pending", "won", "lost", "cancelled"]

    def _load_bets(game_filter: str | None = None, status_filter: str | None = None) -> pd.DataFrame:
        _conn_r = get_connection()
        try:
            _df = _conn_r.execute("""
                SELECT id, saved_at, game, draw_time, combination,
                       label,
                       COALESCE(source, 'bet') AS source,
                       bet_date,
                       COALESCE(status, 'pending') AS status,
                       LEFT(reasoning, 400) AS reasoning_preview,
                       actual_draw,
                       match_count,
                       number_distances,
                       ml_run_id
                FROM saved_combinations
                ORDER BY
                    CASE COALESCE(status,'pending') WHEN 'pending' THEN 0 ELSE 1 END,
                    bet_date ASC NULLS LAST,
                    saved_at DESC
            """).fetchdf()
        finally:
            _conn_r.close()
        if _df.empty:
            return _df
        if game_filter and game_filter != "All":
            _df = _df[_df["game"] == game_filter]
        if status_filter and status_filter != "All":
            _df = _df[_df["status"] == status_filter]
        return _df

    # Ensure table + columns exist for DBs created before bet_date/status were added
    _sv_init_conn = get_connection()
    try:
        for _col, _ctype in [("bet_date", "DATE"), ("status", "VARCHAR(16)"),
                              ("source", "VARCHAR(16)")]:
            try:
                _sv_init_conn.execute(
                    f"ALTER TABLE saved_combinations ADD COLUMN {_col} {_ctype}"
                )
            except Exception:
                pass
    finally:
        _sv_init_conn.close()

    _bets_all = _load_bets()

    if _bets_all.empty:
        st.info(
            "No bets saved yet.  \n"
            "Go to **Probability & AI**, run an AI analysis, "
            "then click **🎯 Save as Bet**."
        )
    else:
        # Summary metrics
        _n_pending = int((_bets_all["status"] == "pending").sum())
        _n_won = int((_bets_all["status"] == "won").sum())
        _n_lost = int((_bets_all["status"] == "lost").sum())
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Total Bets", len(_bets_all))
        _mc2.metric("⏳ Pending", _n_pending)
        _mc3.metric("🏆 Won", _n_won)
        _mc4.metric("❌ Lost", _n_lost)

        st.markdown("---")

        # Filters
        _fc1, _fc2 = st.columns(2)
        with _fc1:
            _sv_games = ["All"] + sorted(_bets_all["game"].unique().tolist())
            _sv_game_filter = st.selectbox(
                "Game", _sv_games,
                format_func=lambda k: GAMES[k].name if k != "All" else "All games",
                key="sv_game_filter",
            )
        with _fc2:
            _sv_status_filter = st.selectbox(
                "Status", ["All"] + _STATUS_OPTIONS,
                format_func=lambda s: f"{_STATUS_EMOJI.get(s, '')} {s.capitalize()}" if s != "All" else "All statuses",
                key="sv_status_filter",
            )

        _bets = _load_bets(_sv_game_filter, _sv_status_filter)

        if _bets.empty:
            st.info("No bets match the selected filters.")
        else:
            for _, _row in _bets.iterrows():
                _gcfg = GAMES.get(str(_row["game"]))
                _gname = _gcfg.name if _gcfg else _row["game"]
                _nums = [int(n) for n in str(_row["combination"]).split("-") if n.strip().isdigit()]
                _dt_str = f" · {_row['draw_time']}" if pd.notna(_row["draw_time"]) and _row["draw_time"] else ""
                _lbl = f" — {_row['label']}" if pd.notna(_row["label"]) and _row["label"] else ""
                _bet_d = str(_row["bet_date"])[:10] if pd.notna(_row.get("bet_date")) and _row.get("bet_date") else "—"
                _status = str(_row.get("status") or "pending")
                _sem = _STATUS_EMOJI.get(_status, "⏳")
                _saved_str = str(_row["saved_at"])[:16]

                _exp_title = (
                    f"{_sem} {_gname}{_dt_str}{_lbl}  |  {_row['combination']}  "
                    f"|  Draw: {_bet_d}  |  Saved: {_saved_str}"
                )

                with st.expander(_exp_title, expanded=(_status == "pending")):
                    # Combination display
                    _combo_md = "  ·  ".join(f"**{n:02d}**" for n in _nums)
                    st.markdown(f"## {_combo_md}")
                    st.caption(
                        f"Game: **{_gname}**{_dt_str}  |  "
                        f"Bet draw: **{_bet_d}**  |  "
                        f"Saved: {_saved_str}"
                    )

                    # Status + result update row
                    _ua, _ub = st.columns([2, 2])
                    with _ua:
                        _new_status = st.selectbox(
                            "Status",
                            _STATUS_OPTIONS,
                            index=_STATUS_OPTIONS.index(_status) if _status in _STATUS_OPTIONS else 0,
                            format_func=lambda s: f"{_STATUS_EMOJI[s]} {s.capitalize()}",
                            key=f"status_sel_{_row['id']}",
                        )
                    with _ub:
                        _new_label = st.text_input(
                            "Note",
                            value=str(_row["label"] or ""),
                            key=f"lbl_{_row['id']}",
                            placeholder="Add a note…",
                        )

                    _bd_col, _save_col = st.columns([2, 1])
                    with _bd_col:
                        _cur_bet_date = None
                        if pd.notna(_row.get("bet_date")) and _row.get("bet_date"):
                            try:
                                _cur_bet_date = pd.to_datetime(_row["bet_date"]).date()
                            except Exception:
                                pass
                        _new_bet_date = st.date_input(
                            "Draw date",
                            value=_cur_bet_date,
                            key=f"bd_{_row['id']}",
                            format="YYYY-MM-DD",
                        )
                    with _save_col:
                        st.write("")
                        st.write("")
                        if st.button("Save changes", key=f"upd_{_row['id']}"):
                            _uc = get_connection()
                            try:
                                _uc.execute(
                                    """UPDATE saved_combinations
                                       SET status = ?, label = ?, bet_date = ?
                                       WHERE id = ?""",
                                    [_new_status, _new_label or None,
                                     _new_bet_date, int(_row["id"])],
                                )
                            finally:
                                _uc.close()
                            st.rerun()

                    # Check vs actual draw on the bet date (or latest draw if no bet date)
                    if _gcfg and _nums:
                        _check_date = None
                        if pd.notna(_row.get("bet_date")) and _row.get("bet_date"):
                            try:
                                _check_date = pd.to_datetime(_row["bet_date"]).date()
                            except Exception:
                                pass

                        if _check_date:
                            _actual = _query(
                                "SELECT draw_date, " + ", ".join(f"n{i}" for i in range(1, _gcfg.picks + 1)) +
                                " FROM draws WHERE game = ? AND draw_date = ? LIMIT 1",
                                [_gcfg.id, _check_date.isoformat()],
                            )
                            _check_label = f"Draw on {_check_date}"
                        else:
                            _actual = _query(
                                "SELECT draw_date, " + ", ".join(f"n{i}" for i in range(1, _gcfg.picks + 1)) +
                                " FROM draws WHERE game = ? ORDER BY draw_date DESC LIMIT 1",
                                [_gcfg.id],
                            )
                            _check_label = "Latest draw"

                        if not _actual.empty:
                            _act_date = _actual.iloc[0]["draw_date"]
                            _act_nums = {
                                int(_actual.iloc[0][f"n{i}"])
                                for i in range(1, _gcfg.picks + 1)
                                if pd.notna(_actual.iloc[0].get(f"n{i}"))
                            }
                            _hits = [n for n in _nums if n in _act_nums]
                            _hit_count = len(_hits)
                            _result_line = " ".join(
                                (f"✅ **{n:02d}**" if n in _act_nums else f"❌ {n:02d}")
                                for n in _nums
                            )
                            st.markdown(
                                f"**{_check_label} ({_act_date}):**  {_result_line}  "
                                f"→  **{_hit_count}/{_gcfg.picks} matched**"
                            )
                            _win_combo_md = "  ·  ".join(f"**{n:02d}**" for n in sorted(_act_nums))
                            st.markdown(f"### 🏅 Winning: {_win_combo_md}")
                            if _hit_count == _gcfg.picks:
                                st.success("🏆 JACKPOT — all numbers matched!")
                            elif _hit_count >= _gcfg.picks - 1:
                                st.info(f"Almost! {_hit_count} of {_gcfg.picks} matched.")

                    # Persisted match result (populated by Refresh Results)
                    _actual_json = _row.get("actual_draw")
                    if pd.notna(_actual_json) and _actual_json:
                        try:
                            import json as _json_sv
                            _actual_sv   = _json_sv.loads(_actual_json)
                            _match_sv    = int(_row.get("match_count") or 0)
                            _dists_sv    = _json_sv.loads(_row.get("number_distances") or "{}")
                            _result_parts = []
                            for _n in _nums:
                                if _dists_sv.get(str(_n), {}).get("hit"):
                                    _result_parts.append(f"✅ **{_n:02d}**")
                                else:
                                    _delta   = _dists_sv.get(str(_n), {}).get("delta", "?")
                                    _closest = _dists_sv.get(str(_n), {}).get("closest_actual", "?")
                                    _result_parts.append(f"❌ {_n:02d}(→{_closest}, Δ{_delta})")
                            st.markdown(
                                "**Saved result:** " + "  ·  ".join(_result_parts)
                                + f"  →  **{_match_sv}/{_gcfg.picks} matched**"
                            )
                            if isinstance(_actual_sv, list) and _actual_sv:
                                _sv_win_md = "  ·  ".join(f"**{n:02d}**" for n in sorted(int(x) for x in _actual_sv))
                                st.markdown(f"### 🏅 Winning: {_sv_win_md}")
                            if _match_sv == _gcfg.picks:
                                st.success("🏆 JACKPOT — all numbers matched!")
                            elif _match_sv >= _gcfg.picks - 1:
                                st.info(f"Near miss! {_match_sv} of {_gcfg.picks} matched.")
                        except Exception:
                            pass

                    if str(_row.get("reasoning_preview") or ""):
                        _rp = str(_row["reasoning_preview"])
                        st.caption(f"💬 {_rp}" + ("…" if len(_rp) >= 400 else ""))

                    st.markdown("---")
                    if st.button("🗑 Delete bet", key=f"del_{_row['id']}", type="secondary"):
                        _dc = get_connection()
                        try:
                            _dc.execute("DELETE FROM saved_combinations WHERE id = ?",
                                        [int(_row["id"])])
                        finally:
                            _dc.close()
                        st.rerun()

        # Export
        st.markdown("---")
        if not _bets_all.empty:
            _export_df = _bets_all.drop(columns=["reasoning_preview"], errors="ignore")
            st.download_button(
                "⬇ Download all bets (CSV)",
                data=_export_df.to_csv(index=False).encode("utf-8"),
                file_name="my_bets.csv",
                mime="text/csv",
            )

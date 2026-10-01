"""DuckDB connection and schema initialisation."""
import os
import duckdb

_DB_PATH = os.environ.get("LOTTO_DB", "data/processed/lotto.duckdb")


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (or create) the DuckDB database and return a connection."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = duckdb.connect(_DB_PATH, read_only=read_only)
    if not read_only:
        _init_schema(conn)
    return conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS draws (
            draw_date       DATE        NOT NULL,
            weekday         VARCHAR(3)  NOT NULL,
            game            VARCHAR(8)  NOT NULL,
            draw_time       VARCHAR(5)  DEFAULT NULL,
            pub_1           SMALLINT    DEFAULT NULL,
            pub_2           SMALLINT    DEFAULT NULL,
            pub_3           SMALLINT    DEFAULT NULL,
            pub_4           SMALLINT    DEFAULT NULL,
            pub_5           SMALLINT    DEFAULT NULL,
            pub_6           SMALLINT    DEFAULT NULL,
            n1              SMALLINT    DEFAULT NULL,
            n2              SMALLINT    DEFAULT NULL,
            n3              SMALLINT    DEFAULT NULL,
            n4              SMALLINT    DEFAULT NULL,
            n5              SMALLINT    DEFAULT NULL,
            n6              SMALLINT    DEFAULT NULL,
            jackpot_php     DOUBLE      DEFAULT NULL,
            notes           TEXT        DEFAULT NULL,
            -- derived columns (populated by clean.py on insert; backfilled by migrate.py)
            draw_sum        SMALLINT    DEFAULT NULL,
            odd_count       TINYINT     DEFAULT NULL,
            high_count      TINYINT     DEFAULT NULL,
            consecutive_cnt TINYINT     DEFAULT NULL,
            number_spread   SMALLINT    DEFAULT NULL,
            is_rollover     BOOLEAN     DEFAULT NULL,
            covid_gap       BOOLEAN     DEFAULT NULL,
            -- winners data (populated by scraper when source provides it)
            winners_jackpot SMALLINT    DEFAULT NULL,
            winners_5match  SMALLINT    DEFAULT NULL,
            winners_4match  SMALLINT    DEFAULT NULL,
            winners_3match  SMALLINT    DEFAULT NULL,
            PRIMARY KEY (draw_date, game, draw_time)
        )
    """)

    # Migrate existing DBs that pre-date the derived columns
    try:
        from src.lotto.migrate import add_derived_columns
        add_derived_columns(conn)
    except Exception:
        pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS scrape_log (
            id          INTEGER PRIMARY KEY,
            url         TEXT        NOT NULL,
            fetched_at  TIMESTAMP   NOT NULL,
            status      SMALLINT    NOT NULL,
            rows_parsed SMALLINT    DEFAULT NULL,
            cache_path  TEXT        DEFAULT NULL
        )
    """)

    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS scrape_log_seq START 1
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_combinations (
            id          INTEGER PRIMARY KEY,
            saved_at    TIMESTAMP   NOT NULL DEFAULT NOW(),
            game        VARCHAR(8)  NOT NULL,
            draw_time   VARCHAR(5)  DEFAULT NULL,
            combination VARCHAR(60) NOT NULL,
            reasoning   TEXT        DEFAULT NULL,
            label       TEXT        DEFAULT NULL,
            source      VARCHAR(16) DEFAULT 'bet',
            bet_date    DATE        DEFAULT NULL,
            status      VARCHAR(16) DEFAULT 'pending'
        )
    """)

    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS saved_combinations_seq START 1
    """)

    # Migrate existing saved_combinations tables that predate bet_date/status columns
    for _sc_col, _sc_type in [
        ("bet_date",         "DATE"),
        ("status",           "VARCHAR(16)"),
        ("source",           "VARCHAR(16)"),
        ("ml_run_id",        "INTEGER"),
        ("actual_draw",      "VARCHAR"),
        ("match_count",      "INTEGER"),
        ("number_distances", "VARCHAR"),
    ]:
        try:
            conn.execute(
                f"ALTER TABLE saved_combinations ADD COLUMN {_sc_col} {_sc_type}"
            )
        except Exception:
            pass

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ml_prediction_runs (
            id                 INTEGER PRIMARY KEY,
            run_at             TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
            game               VARCHAR(8)  NOT NULL,
            draw_time          VARCHAR(10) DEFAULT NULL,
            as_of_date         DATE        DEFAULT NULL,
            n_training_draws   INTEGER     DEFAULT NULL,
            model_version      VARCHAR(16) DEFAULT NULL,
            feature_importance VARCHAR     DEFAULT NULL,
            number_probs       VARCHAR     DEFAULT NULL,
            top_combinations   VARCHAR     DEFAULT NULL,
            consensus_core     VARCHAR     DEFAULT NULL,
            runner_ups         VARCHAR     DEFAULT NULL,
            ai_combination     VARCHAR     DEFAULT NULL,
            ai_reasoning       TEXT        DEFAULT NULL,
            ai_prompt_text     TEXT        DEFAULT NULL,
            brier_score        DOUBLE      DEFAULT NULL,
            ece                DOUBLE      DEFAULT NULL
        )
    """)

    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS ml_prediction_runs_seq START 1
    """)


def save_ml_run(
    game: str,
    draw_time: "str | None",
    as_of_date: str,
    n_training_draws: int,
    model_version: str,
    feature_importance: "dict | None" = None,
    number_probs: "dict | None" = None,
    top_combinations: "list | None" = None,
    consensus_core: "list | None" = None,
    runner_ups: "list | None" = None,
    ai_combination: "list | None" = None,
    ai_reasoning: "str | None" = None,
    ai_prompt_text: "str | None" = None,
    brier_score: "float | None" = None,
    ece: "float | None" = None,
) -> int:
    """Persist one ML prediction run. Returns the new row id."""
    import json
    conn = get_connection()
    try:
        row = conn.execute(
            """
            INSERT INTO ml_prediction_runs
                (id, run_at, game, draw_time, as_of_date, n_training_draws,
                 model_version, feature_importance, number_probs, top_combinations,
                 consensus_core, runner_ups, ai_combination,
                 ai_reasoning, ai_prompt_text, brier_score, ece)
            VALUES
                (nextval('ml_prediction_runs_seq'), NOW(),
                 ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            [
                game, draw_time, as_of_date, n_training_draws, model_version,
                json.dumps(feature_importance) if feature_importance is not None else None,
                json.dumps({str(k): v for k, v in number_probs.items()})
                    if number_probs is not None else None,
                json.dumps(top_combinations) if top_combinations is not None else None,
                json.dumps(consensus_core) if consensus_core is not None else None,
                json.dumps(runner_ups) if runner_ups is not None else None,
                json.dumps(ai_combination) if ai_combination is not None else None,
                ai_reasoning, ai_prompt_text, brier_score, ece,
            ],
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def compare_with_actual(game: "str | None" = None) -> int:
    """
    For every saved_combinations row whose bet_date matches a draws record
    and actual_draw is NULL, compute match_count and number_distances and
    update the row. Returns the count of rows updated.
    """
    import json
    conn = get_connection()
    try:
        query = """
            SELECT sc.id, sc.game, sc.draw_time, sc.combination,
                   d.n1, d.n2, d.n3, d.n4, d.n5, d.n6
            FROM saved_combinations sc
            JOIN draws d ON (
                d.draw_date = sc.bet_date
                AND d.game = sc.game
                AND (
                    sc.draw_time IS NULL
                    OR d.draw_time = sc.draw_time
                    OR (d.draw_time = '14:00' AND sc.draw_time = '2:00 PM')
                    OR (d.draw_time = '17:00' AND sc.draw_time = '5:00 PM')
                    OR (d.draw_time = '21:00' AND sc.draw_time = '9:00 PM')
                    OR (sc.draw_time = '14:00' AND d.draw_time = '2:00 PM')
                    OR (sc.draw_time = '17:00' AND d.draw_time = '5:00 PM')
                    OR (sc.draw_time = '21:00' AND d.draw_time = '9:00 PM')
                )
            )
            WHERE sc.actual_draw IS NULL
            AND sc.bet_date IS NOT NULL
        """
        params: list = []
        if game:
            query += " AND sc.game = ?"
            params.append(game)

        rows = conn.execute(query, params).fetchall()
        updated = 0

        for row in rows:
            sc_id, _, _, combination = row[0], row[1], row[2], row[3]
            raw_cols = row[4:]

            actual_nums = [int(x) for x in raw_cols if x is not None]
            if not actual_nums:
                continue

            ai_nums = [
                int(n) for n in str(combination).split("-")
                if n.strip().isdigit()
            ]
            actual_set = set(actual_nums)
            match_count = sum(1 for n in ai_nums if n in actual_set)

            number_distances: dict = {}
            for n in ai_nums:
                if n in actual_set:
                    number_distances[str(n)] = {"closest_actual": n, "delta": 0, "hit": True}
                else:
                    closest = min(actual_nums, key=lambda x: abs(x - n))
                    number_distances[str(n)] = {
                        "closest_actual": closest,
                        "delta": abs(closest - n),
                        "hit": False,
                    }

            conn.execute(
                """UPDATE saved_combinations
                   SET actual_draw      = ?,
                       match_count      = ?,
                       number_distances = ?
                   WHERE id = ?""",
                [json.dumps(actual_nums), match_count, json.dumps(number_distances), sc_id],
            )
            updated += 1

        return updated
    finally:
        conn.close()


def load_prior_ml_runs(
    game: str,
    draw_time: "str | None" = None,
    n: int = 5,
) -> "list[dict]":
    """
    Return up to N completed feedback records: ML runs linked to a
    saved_combinations row that already has actual_draw populated.
    """
    import json
    conn = get_connection()
    try:
        query = """
            SELECT
                r.run_at,
                r.consensus_core,
                r.feature_importance,
                r.ai_combination,
                sc.actual_draw,
                sc.match_count,
                sc.number_distances
            FROM ml_prediction_runs r
            JOIN saved_combinations sc ON sc.ml_run_id = r.id
            WHERE r.game = ?
            AND r.ai_combination IS NOT NULL
            AND sc.actual_draw IS NOT NULL
        """
        params: list = [game]

        if draw_time:
            query += """
              AND (
                r.draw_time = ?
                OR r.draw_time IS NULL
                OR (r.draw_time = '2:00 PM'  AND ? = '14:00')
                OR (r.draw_time = '14:00'    AND ? = '2:00 PM')
                OR (r.draw_time = '5:00 PM'  AND ? = '17:00')
                OR (r.draw_time = '17:00'    AND ? = '5:00 PM')
                OR (r.draw_time = '9:00 PM'  AND ? = '21:00')
                OR (r.draw_time = '21:00'    AND ? = '9:00 PM')
              )
            """
            params.extend([draw_time] * 8)

        query += " ORDER BY r.run_at DESC LIMIT ?"
        params.append(n)

        rows = conn.execute(query, params).fetchall()
        result: list = []
        for row in rows:
            (run_at, consensus_raw, feat_raw,
             ai_combo_raw, actual_raw, match_count, distances_raw) = row

            ai_combo  = json.loads(ai_combo_raw)  if ai_combo_raw  else []
            actual    = json.loads(actual_raw)     if actual_raw    else []
            distances = json.loads(distances_raw)  if distances_raw else {}
            consensus = json.loads(consensus_raw)  if consensus_raw else []
            feat      = json.loads(feat_raw)       if feat_raw      else {}

            total_dist = sum(
                v["delta"] for v in distances.values() if not v.get("hit")
            )
            top_feats = dict(
                sorted(feat.items(), key=lambda x: x[1], reverse=True)[:3]
            )

            result.append({
                "run_date":            str(run_at)[:10],
                "ai_combination":      ai_combo,
                "actual_draw":         actual,
                "match_count":         match_count or 0,
                "total_distance":      total_dist,
                "number_distances":    distances,
                "consensus_core_then": consensus,
                "top_features_then":   top_feats,
            })

        return result
    finally:
        conn.close()

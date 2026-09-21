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
            draw_date   DATE        NOT NULL,
            weekday     VARCHAR(3)  NOT NULL,
            game        VARCHAR(8)  NOT NULL,
            draw_time   VARCHAR(5)  DEFAULT NULL,
            pub_1       SMALLINT    DEFAULT NULL,
            pub_2       SMALLINT    DEFAULT NULL,
            pub_3       SMALLINT    DEFAULT NULL,
            pub_4       SMALLINT    DEFAULT NULL,
            pub_5       SMALLINT    DEFAULT NULL,
            pub_6       SMALLINT    DEFAULT NULL,
            n1          SMALLINT    DEFAULT NULL,
            n2          SMALLINT    DEFAULT NULL,
            n3          SMALLINT    DEFAULT NULL,
            n4          SMALLINT    DEFAULT NULL,
            n5          SMALLINT    DEFAULT NULL,
            n6          SMALLINT    DEFAULT NULL,
            jackpot_php DOUBLE      DEFAULT NULL,
            notes       TEXT        DEFAULT NULL,
            PRIMARY KEY (draw_date, game, draw_time)
        )
    """)

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

"""DuckDB warehouse: posts, hourly sentiment, prices, news and the joined modelling view.

DuckDB (embedded, columnar, zero-config) is chosen over Postgres so a reviewer can reproduce the
whole pipeline with `pip install` and no server. The join is done in SQL so the alignment rule
(hourly UTC bars, posts bucketed by created_utc floor-hour) is explicit and inspectable.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from .config import DB_PATH

DDL = """
CREATE TABLE IF NOT EXISTS prices_1h (time TIMESTAMPTZ PRIMARY KEY, open DOUBLE, high DOUBLE,
    low DOUBLE, close DOUBLE);
CREATE TABLE IF NOT EXISTS sent_hourly (time TIMESTAMPTZ PRIMARY KEY, n_posts DOUBLE, n_human DOUBLE,
    bot_share DOUBLE, sent_mean DOUBLE, sent_std DOUBLE, bull_mean DOUBLE, bear_mean DOUBLE,
    sent_w DOUBLE, h_sent_mean DOUBLE, h_sent_std DOUBLE, h_bull_mean DOUBLE, h_bear_mean DOUBLE,
    h_sent_w DOUBLE, h_bullishness DOUBLE, bullishness DOUBLE, authors DOUBLE, source VARCHAR);
CREATE TABLE IF NOT EXISTS news (url VARCHAR PRIMARY KEY, title VARCHAR, content VARCHAR,
    published TIMESTAMPTZ, query VARCHAR, source VARCHAR, sent DOUBLE);
CREATE OR REPLACE VIEW model_input AS
    SELECT p.time, p.open, p.high, p.low, p.close, s.* EXCLUDE (time)
    FROM prices_1h p LEFT JOIN sent_hourly s ON s.time = p.time
    ORDER BY p.time;
"""


def connect(path=DB_PATH):
    con = duckdb.connect(str(path))
    con.execute(DDL)
    return con


def upsert(con, table: str, df: pd.DataFrame, key: str = "time"):
    df = df.reset_index() if key not in df.columns else df
    cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    df = df[[c for c in cols if c in df.columns]]
    con.register("tmp_df", df)
    con.execute(f"DELETE FROM {table} WHERE {key} IN (SELECT {key} FROM tmp_df)")
    con.execute(f"INSERT INTO {table} ({', '.join(df.columns)}) SELECT * FROM tmp_df")
    con.unregister("tmp_df")


def load_model_input(con, start=None, end=None) -> pd.DataFrame:
    q = "SELECT * FROM model_input"
    if start:
        q += f" WHERE time >= TIMESTAMPTZ '{start}'" + (f" AND time <= TIMESTAMPTZ '{end}'" if end else "")
    df = con.execute(q).df()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")

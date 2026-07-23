import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
  pair TEXT NOT NULL,
  interval_minutes INTEGER NOT NULL,
  timestamp INTEGER NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  vwap REAL NOT NULL,
  volume REAL NOT NULL,
  trades INTEGER NOT NULL,
  PRIMARY KEY (pair, interval_minutes, timestamp)
);
CREATE TABLE IF NOT EXISTS market_trades (
  pair TEXT NOT NULL,
  trade_id INTEGER NOT NULL,
  timestamp REAL NOT NULL,
  price REAL NOT NULL,
  volume REAL NOT NULL,
  PRIMARY KEY (pair, trade_id)
);
CREATE INDEX IF NOT EXISTS idx_market_trades_timestamp
  ON market_trades(pair, timestamp);
CREATE TABLE IF NOT EXISTS history_hour_accumulators (
  pair TEXT NOT NULL,
  interval_minutes INTEGER NOT NULL,
  timestamp INTEGER NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL NOT NULL,
  price_volume REAL NOT NULL,
  trades INTEGER NOT NULL,
  first_trade_timestamp REAL NOT NULL,
  first_trade_id INTEGER NOT NULL,
  last_trade_timestamp REAL NOT NULL,
  last_trade_id INTEGER NOT NULL,
  PRIMARY KEY (pair, interval_minutes, timestamp)
);
CREATE TABLE IF NOT EXISTS history_sync_state (
  pair TEXT NOT NULL,
  interval_minutes INTEGER NOT NULL,
  cursor TEXT NOT NULL,
  status TEXT NOT NULL,
  batches INTEGER NOT NULL,
  unique_trades INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (pair, interval_minutes)
);
CREATE TABLE IF NOT EXISTS paper_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  cash REAL NOT NULL,
  asset REAL NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pair TEXT NOT NULL,
  interval_minutes INTEGER NOT NULL,
  timestamp INTEGER NOT NULL,
  action TEXT NOT NULL,
  probability REAL NOT NULL,
  price REAL NOT NULL,
  equity REAL NOT NULL,
  locked INTEGER NOT NULL DEFAULT 0,
  reason TEXT NOT NULL,
  UNIQUE (pair, interval_minutes, timestamp)
);
CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
  timestamp INTEGER PRIMARY KEY,
  equity REAL NOT NULL,
  cash REAL NOT NULL,
  asset REAL NOT NULL,
  drawdown REAL NOT NULL,
  daily_return REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp INTEGER NOT NULL,
  side TEXT NOT NULL,
  price REAL NOT NULL,
  quantity REAL NOT NULL,
  fee REAL NOT NULL,
  probability REAL NOT NULL,
  reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  created_at INTEGER NOT NULL,
  config_json TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  model_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  UNIQUE(config_hash, created_at)
);
CREATE TABLE IF NOT EXISTS experiment_folds (
  experiment_id TEXT NOT NULL,
  fold_id INTEGER NOT NULL,
  train_start INTEGER NOT NULL,
  train_end INTEGER NOT NULL,
  test_start INTEGER NOT NULL,
  test_end INTEGER NOT NULL,
  train_rows INTEGER NOT NULL,
  test_rows INTEGER NOT NULL,
  model_hash TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  PRIMARY KEY (experiment_id, fold_id),
  FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE TABLE IF NOT EXISTS experiment_predictions (
  experiment_id TEXT NOT NULL,
  fold_id INTEGER NOT NULL,
  timestamp INTEGER NOT NULL,
  close REAL NOT NULL,
  target INTEGER NOT NULL,
  target_3way INTEGER NOT NULL,
  probability REAL NOT NULL,
  probability_buy_3way REAL NOT NULL,
  train_end INTEGER NOT NULL,
  PRIMARY KEY (experiment_id, timestamp),
  FOREIGN KEY (experiment_id, fold_id) REFERENCES experiment_folds(experiment_id, fold_id)
);
"""


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def upsert_candles(path: Path, df: pd.DataFrame, pair: str, interval: int) -> int:
    rows = [
        (
            pair,
            interval,
            int(r.timestamp),
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            float(r.vwap),
            float(r.volume),
            int(r.trades),
        )
        for r in df.itertuples(index=False)
    ]
    with connect(path) as con:
        before = con.total_changes
        con.executemany(
            """
            INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pair, interval_minutes, timestamp) DO UPDATE SET
              open=excluded.open, high=excluded.high, low=excluded.low,
              close=excluded.close, vwap=excluded.vwap,
              volume=excluded.volume, trades=excluded.trades
            """,
            rows,
        )
        return con.total_changes - before


def load_candles(path: Path, pair: str, interval: int) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(
            """SELECT timestamp, open, high, low, close, vwap, volume, trades
               FROM candles WHERE pair=? AND interval_minutes=? ORDER BY timestamp""",
            con,
            params=(pair, interval),
        )

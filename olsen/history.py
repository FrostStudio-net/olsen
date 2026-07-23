import math
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .db import connect, load_candles, upsert_candles
from .kraken import KrakenPublicClient


@dataclass
class HistorySummary:
    first_candle: int
    last_candle: int
    candles: int
    missing_candles: int
    database_size: int

    @staticmethod
    def format_timestamp(timestamp: int) -> str:
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _set_sync_status(
    db_path: Path,
    pair: str,
    interval: int,
    status: str,
    cursor: str | None = None,
) -> None:
    with connect(db_path) as con:
        existing = con.execute(
            """SELECT cursor, batches, unique_trades FROM history_sync_state
               WHERE pair=? AND interval_minutes=?""",
            (pair, interval),
        ).fetchone()
        saved_cursor, batches, trades = existing if existing else ("0", 0, 0)
        con.execute(
            """INSERT INTO history_sync_state VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(pair, interval_minutes) DO UPDATE SET
                 cursor=excluded.cursor, status=excluded.status,
                 batches=excluded.batches, unique_trades=excluded.unique_trades,
                 updated_at=excluded.updated_at""",
            (pair, interval, cursor or saved_cursor, status, batches, trades, int(time.time())),
        )


def _sync_cursor(db_path: Path, pair: str, interval: int) -> str:
    with connect(db_path) as con:
        row = con.execute(
            "SELECT cursor FROM history_sync_state WHERE pair=? AND interval_minutes=?",
            (pair, interval),
        ).fetchone()
    return str(row[0]) if row else "0"


def _aggregate_new_trades(
    con: sqlite3.Connection,
    trades: pd.DataFrame,
    pair: str,
    interval: int,
) -> int:
    seconds = interval * 60
    buckets: dict[int, dict[str, float | int]] = {}
    inserted = 0
    for row in trades.itertuples(index=False):
        trade_id = int(row.trade_id)
        timestamp, price, volume = float(row.timestamp), float(row.price), float(row.volume)
        cursor = con.execute(
            "INSERT OR IGNORE INTO market_trades VALUES (?, ?, ?, ?, ?)",
            (pair, trade_id, timestamp, price, volume),
        )
        if cursor.rowcount == 0:
            continue
        inserted += 1
        bucket = int(timestamp // seconds * seconds)
        aggregate = buckets.get(bucket)
        if aggregate is None:
            buckets[bucket] = {
                "open": price, "high": price, "low": price, "close": price,
                "volume": volume, "price_volume": price * volume, "trades": 1,
                "first_time": timestamp, "first_id": trade_id,
                "last_time": timestamp, "last_id": trade_id,
            }
            continue
        aggregate["high"] = max(float(aggregate["high"]), price)
        aggregate["low"] = min(float(aggregate["low"]), price)
        aggregate["volume"] = float(aggregate["volume"]) + volume
        aggregate["price_volume"] = float(aggregate["price_volume"]) + price * volume
        aggregate["trades"] = int(aggregate["trades"]) + 1
        first_key = (float(aggregate["first_time"]), int(aggregate["first_id"]))
        last_key = (float(aggregate["last_time"]), int(aggregate["last_id"]))
        if (timestamp, trade_id) < first_key:
            aggregate.update(open=price, first_time=timestamp, first_id=trade_id)
        if (timestamp, trade_id) > last_key:
            aggregate.update(close=price, last_time=timestamp, last_id=trade_id)
    con.executemany(
        """INSERT INTO history_hour_accumulators VALUES
             (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(pair, interval_minutes, timestamp) DO UPDATE SET
             open=CASE
               WHEN (excluded.first_trade_timestamp < first_trade_timestamp)
                 OR (excluded.first_trade_timestamp = first_trade_timestamp
                     AND excluded.first_trade_id < first_trade_id)
               THEN excluded.open ELSE open END,
             high=MAX(high, excluded.high), low=MIN(low, excluded.low),
             close=CASE
               WHEN (excluded.last_trade_timestamp > last_trade_timestamp)
                 OR (excluded.last_trade_timestamp = last_trade_timestamp
                     AND excluded.last_trade_id > last_trade_id)
               THEN excluded.close ELSE close END,
             volume=volume + excluded.volume,
             price_volume=price_volume + excluded.price_volume,
             trades=trades + excluded.trades,
             first_trade_timestamp=MIN(first_trade_timestamp, excluded.first_trade_timestamp),
             first_trade_id=CASE
               WHEN excluded.first_trade_timestamp <= first_trade_timestamp
               THEN MIN(first_trade_id, excluded.first_trade_id) ELSE first_trade_id END,
             last_trade_timestamp=MAX(last_trade_timestamp, excluded.last_trade_timestamp),
             last_trade_id=CASE
               WHEN excluded.last_trade_timestamp >= last_trade_timestamp
               THEN MAX(last_trade_id, excluded.last_trade_id) ELSE last_trade_id END""",
        [
            (
                pair, interval, bucket, values["open"], values["high"], values["low"],
                values["close"], values["volume"], values["price_volume"], values["trades"],
                values["first_time"], values["first_id"], values["last_time"], values["last_id"],
            )
            for bucket, values in buckets.items()
        ],
    )
    return inserted


def _store_batch(
    db_path: Path,
    trades: pd.DataFrame,
    pair: str,
    interval: int,
    next_cursor: str,
) -> int:
    with connect(db_path) as con:
        con.execute("BEGIN IMMEDIATE")
        inserted = _aggregate_new_trades(con, trades, pair, interval)
        con.execute(
            """UPDATE history_sync_state SET cursor=?, status='running',
               batches=batches+1, unique_trades=unique_trades+?, updated_at=?
               WHERE pair=? AND interval_minutes=?""",
            (next_cursor, inserted, int(time.time()), pair, interval),
        )
    return inserted


def _materialize_completed_candles(
    db_path: Path, pair: str, interval: int, now: float | None = None
) -> None:
    current_bucket = int((time.time() if now is None else now) // (interval * 60) * (interval * 60))
    with connect(db_path) as con:
        con.execute(
            """INSERT INTO candles
               SELECT pair, interval_minutes, timestamp, open, high, low, close,
                      CASE WHEN volume > 0 THEN price_volume / volume ELSE close END,
                      volume, trades
               FROM history_hour_accumulators
               WHERE pair=? AND interval_minutes=? AND timestamp < ?
               ON CONFLICT(pair, interval_minutes, timestamp) DO UPDATE SET
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, vwap=excluded.vwap,
                 volume=excluded.volume, trades=excluded.trades""",
            (pair, interval, current_bucket),
        )


def verify_history(db_path: Path, pair: str, interval: int) -> HistorySummary:
    candles = load_candles(db_path, pair, interval)
    if candles.empty:
        raise ValueError("History sync produced no candles.")
    seconds = interval * 60
    timestamps = candles["timestamp"].astype(int)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Candle timestamps are duplicated or out of order.")
    if (timestamps % seconds != 0).any():
        raise ValueError("Candle timestamps are not aligned to the configured interval.")
    numeric = candles[["open", "high", "low", "close", "vwap", "volume"]]
    if not numeric.map(math.isfinite).all().all():
        raise ValueError("Candle history contains non-finite numeric values.")
    invalid = (
        (candles[["open", "high", "low", "close", "vwap"]] <= 0).any(axis=1)
        | (candles["volume"] < 0)
        | (candles["trades"] < 0)
        | (candles["high"] < candles[["open", "close", "low"]].max(axis=1))
        | (candles["low"] > candles[["open", "close", "high"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError(f"Candle history contains {int(invalid.sum())} invalid OHLCV rows.")
    first, last, count = int(timestamps.iloc[0]), int(timestamps.iloc[-1]), len(candles)
    expected = (last - first) // seconds + 1
    return HistorySummary(first, last, count, int(expected - count), db_path.stat().st_size)


def sync_history(
    db_path: Path,
    pair: str,
    rest_pair: str,
    interval: int,
    request_delay: float = 1.05,
    client: KrakenPublicClient | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> HistorySummary:
    """Resume Kraken trade history, aggregate it, reconcile REST OHLC, and verify."""
    if interval != 60:
        raise ValueError("Automatic history sync currently supports the BTC/EUR 60-minute market.")
    owned_client = client is None
    public_client = client or KrakenPublicClient(timeout=30.0)
    _set_sync_status(db_path, pair, interval, "running")
    cursor = _sync_cursor(db_path, pair, interval)
    batch_number, consecutive_failures = 0, 0
    try:
        while True:
            try:
                trades, next_cursor = public_client.fetch_trades(rest_pair, cursor, count=1000)
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                if consecutive_failures > 8:
                    raise
                sleep(min(60.0, 2.0 ** consecutive_failures))
                continue
            inserted = _store_batch(db_path, trades, pair, interval, next_cursor)
            batch_number += 1
            if progress is not None:
                progress(batch_number, inserted, next_cursor)
            if trades.empty or next_cursor == cursor:
                cursor = next_cursor
                break
            cursor = next_cursor
            if request_delay:
                sleep(request_delay)
        _materialize_completed_candles(db_path, pair, interval)
        recent = public_client.fetch_ohlc(rest_pair, interval)
        upsert_candles(db_path, recent, pair, interval)
        summary = verify_history(db_path, pair, interval)
        _set_sync_status(db_path, pair, interval, "complete", cursor)
        summary.database_size = db_path.stat().st_size
        return summary
    except BaseException:
        _set_sync_status(db_path, pair, interval, "interrupted")
        raise
    finally:
        if owned_client:
            public_client.close()

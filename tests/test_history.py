import sqlite3

import numpy as np
import pandas as pd
import pytest

from olsen.history import sync_history, verify_history


def _trades(start_id: int, count: int, start_time: float) -> pd.DataFrame:
    ids = np.arange(start_id, start_id + count)
    return pd.DataFrame({
        "price": 100 + ids * 0.01,
        "volume": 0.1,
        "timestamp": start_time + np.arange(count),
        "side": "b",
        "order_type": "m",
        "misc": "",
        "trade_id": ids,
    })


class InterruptedClient:
    def __init__(self):
        self.calls = 0

    def fetch_trades(self, pair, since, count):
        self.calls += 1
        if self.calls == 1:
            assert since == "0"
            return _trades(1, 1000, 1_577_836_801.0), "checkpoint-1"
        raise KeyboardInterrupt


class ResumeClient:
    def __init__(self):
        self.seen_cursor = None
        self.first_cursor = None

    def fetch_trades(self, pair, since, count):
        self.seen_cursor = since
        if self.first_cursor is None:
            self.first_cursor = since
        if since == "checkpoint-1":
            return _trades(1001, 1, 1_577_840_401.0), "checkpoint-2"
        return _trades(1002, 0, 1_577_840_402.0), "checkpoint-2"

    def fetch_ohlc(self, pair, interval):
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "trades"]
        )


class ReplayClient(ResumeClient):
    def fetch_trades(self, pair, since, count):
        if since == "checkpoint-2":
            return _trades(1001, 1, 1_577_840_401.0), "checkpoint-3"
        return _trades(1002, 0, 1_577_840_402.0), "checkpoint-3"


def test_history_sync_resumes_without_duplicates_and_verifies_integrity(tmp_path):
    db_path = tmp_path / "history.db"
    with pytest.raises(KeyboardInterrupt):
        sync_history(
            db_path, "XBT/EUR", "XBTEUR", 60, request_delay=0,
            client=InterruptedClient(), sleep=lambda _: None,
        )

    resume = ResumeClient()
    summary = sync_history(
        db_path, "XBT/EUR", "XBTEUR", 60, request_delay=0,
        client=resume, sleep=lambda _: None,
    )

    assert resume.first_cursor == "checkpoint-1"
    assert summary.candles == 2
    assert summary.missing_candles == 0
    repeated = sync_history(
        db_path, "XBT/EUR", "XBTEUR", 60, request_delay=0,
        client=ReplayClient(), sleep=lambda _: None,
    )
    assert repeated.candles == 2
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM market_trades").fetchone()[0] == 1001
        assert con.execute("SELECT status FROM history_sync_state").fetchone()[0] == "complete"


def test_history_verifier_rejects_misaligned_candles(tmp_path):
    db_path = tmp_path / "invalid.db"
    with sqlite3.connect(db_path) as con:
        con.execute(
            """CREATE TABLE candles (
               pair TEXT, interval_minutes INTEGER, timestamp INTEGER, open REAL,
               high REAL, low REAL, close REAL, vwap REAL, volume REAL, trades INTEGER)"""
        )
        con.execute(
            "INSERT INTO candles VALUES ('XBT/EUR', 60, 101, 1, 1, 1, 1, 1, 1, 1)"
        )

    with pytest.raises(ValueError, match="aligned"):
        verify_history(db_path, "XBT/EUR", 60)

import asyncio
import json
import time
from pathlib import Path

import pandas as pd

from .db import upsert_candles
from .kraken import KrakenPublicClient

WS_URL = "wss://ws.kraken.com/v2"


def candle_from_message(item: dict) -> dict:
    timestamp = int(pd.Timestamp(item["interval_begin"]).timestamp())
    return {
        "timestamp": timestamp,
        "open": float(item["open"]),
        "high": float(item["high"]),
        "low": float(item["low"]),
        "close": float(item["close"]),
        "vwap": float(item["vwap"]),
        "volume": float(item["volume"]),
        "trades": int(item["trades"]),
    }


def reconcile_rest(db_path: Path, pair: str, rest_pair: str, interval: int) -> int:
    client = KrakenPublicClient()
    try:
        candles = client.fetch_ohlc(rest_pair, interval)
    finally:
        client.close()
    return upsert_candles(db_path, candles, pair, interval)


async def collect_ohlc(
    db_path: Path,
    pair: str,
    interval: int,
    rest_pair: str,
    reconcile_seconds: int = 3600,
) -> None:
    """Collect public v2 OHLC updates; persist only candles whose interval has closed."""
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Install Olsen with the 'services' extra for WebSocket collection.") from exc
    last_reconciliation = 0.0
    async for websocket in websockets.connect(WS_URL, ping_interval=20, ping_timeout=20):
        try:
            await websocket.send(json.dumps({
                "method": "subscribe",
                "params": {"channel": "ohlc", "symbol": [pair.replace("XBT", "BTC")],
                           "interval": interval, "snapshot": True},
            }))
            async for raw in websocket:
                payload = json.loads(raw)
                if payload.get("channel") != "ohlc":
                    continue
                current_bucket = int(time.time() // (interval * 60) * (interval * 60))
                rows = [candle_from_message(item) for item in payload.get("data", [])]
                completed = [row for row in rows if row["timestamp"] < current_bucket]
                if completed:
                    upsert_candles(db_path, pd.DataFrame(completed), pair, interval)
                if time.monotonic() - last_reconciliation >= reconcile_seconds:
                    await asyncio.to_thread(reconcile_rest, db_path, pair, rest_pair, interval)
                    last_reconciliation = time.monotonic()
        except Exception:
            await asyncio.sleep(2)
            continue


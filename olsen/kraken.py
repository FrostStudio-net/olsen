import time
from typing import Any

import httpx
import pandas as pd

BASE_URL = "https://api.kraken.com/0/public"


class KrakenPublicClient:
    def __init__(self, timeout: float = 20.0) -> None:
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": "olsen/0.2"})

    def close(self) -> None:
        self.client.close()

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self.client.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Kraken API error: {payload['error']}")
        return payload["result"]

    def fetch_ohlc(self, pair: str, interval: int, since: int | None = None) -> pd.DataFrame:
        params: dict[str, Any] = {"pair": pair, "interval": interval}
        if since is not None:
            params["since"] = since
        result = self._get("OHLC", params)
        key = next(k for k in result if k != "last")
        rows = result[key]
        columns = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "trades"]
        df = pd.DataFrame(rows, columns=columns)
        for c in columns:
            df[c] = pd.to_numeric(df[c])
        # Kraken includes the current, not-yet-committed candle; exclude it.
        current_bucket = int(time.time() // (interval * 60) * (interval * 60))
        return df[df["timestamp"] < current_bucket].reset_index(drop=True)

from fastapi.testclient import TestClient

from olsen.dashboard import create_app
from olsen.db import connect
from olsen.websocket import candle_from_message


def test_dashboard_is_read_only_and_reports_live_disabled(tmp_path):
    db_path = tmp_path / "dashboard.db"
    with connect(db_path):
        pass
    client = TestClient(create_app(db_path))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "mode": "read-only",
        "live_execution": "disabled",
    }


def test_websocket_candle_uses_interval_begin():
    candle = candle_from_message({
        "interval_begin": "2023-11-14T22:00:00Z",
        "open": 100,
        "high": 102,
        "low": 99,
        "close": 101,
        "vwap": 100.5,
        "volume": 2.5,
        "trades": 12,
    })

    assert candle["timestamp"] == 1_699_999_200
    assert candle["close"] == 101.0

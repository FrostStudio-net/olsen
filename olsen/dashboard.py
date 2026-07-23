import json
import sqlite3
from pathlib import Path


def _readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def create_app(db_path: Path):
    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:
        raise RuntimeError("Install Olsen with the 'services' extra for the dashboard API.") from exc

    app = FastAPI(title="Olsen read-only dashboard", version="0.2.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "read-only", "live_execution": "disabled"}

    @app.get("/candles")
    def candles(limit: int = Query(200, ge=1, le=5000)) -> list[dict]:
        with _readonly_connection(db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM candles ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    @app.get("/experiments")
    def experiments() -> list[dict]:
        with _readonly_connection(db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        return [{**dict(row), "config": json.loads(row["config_json"])} for row in rows]

    @app.get("/experiments/{experiment_id}/predictions")
    def predictions(experiment_id: str, limit: int = Query(1000, ge=1, le=10000)) -> list[dict]:
        with _readonly_connection(db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """SELECT * FROM experiment_predictions WHERE experiment_id=?
                   ORDER BY timestamp LIMIT ?""",
                (experiment_id, limit),
            ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Experiment not found or has no predictions")
        return [dict(row) for row in rows]

    @app.get("/paper")
    def paper() -> dict:
        with _readonly_connection(db_path) as con:
            con.row_factory = sqlite3.Row
            state = con.execute("SELECT * FROM paper_state WHERE id=1").fetchone()
            latest = con.execute(
                "SELECT * FROM paper_equity_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return {"state": dict(state) if state else None, "latest": dict(latest) if latest else None}

    return app

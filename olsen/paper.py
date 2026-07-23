import time
from dataclasses import dataclass
from pathlib import Path

from .db import connect
from .features import FEATURE_COLUMNS
from .model import positive_probability


@dataclass
class PaperDecision:
    action: str
    probability: float
    price: float
    cash: float
    asset: float
    equity: float
    locked: bool = False
    reason: str = ""
    duplicate: bool = False


def _existing_decision(con, pair: str, interval: int, timestamp: int):
    return con.execute(
        """SELECT action, probability, price, equity, locked, reason
           FROM paper_decisions WHERE pair=? AND interval_minutes=? AND timestamp=?""",
        (pair, interval, timestamp),
    ).fetchone()


def execute_paper_step(
    db_path: Path,
    feature_row,
    model,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    buy_threshold: float,
    sell_threshold: float,
    max_allocation: float,
    pair: str = "XBT/EUR",
    interval_minutes: int = 60,
    daily_loss_limit: float = 0.03,
    max_drawdown_limit: float = 0.15,
) -> PaperDecision:
    if len(feature_row) != 1:
        raise ValueError("Paper execution requires exactly one completed candle.")
    probability = float(positive_probability(model, feature_row[FEATURE_COLUMNS])[0])
    price, timestamp = float(feature_row.iloc[0]["close"]), int(feature_row.iloc[0]["timestamp"])
    current_bucket = int(time.time() // (interval_minutes * 60) * (interval_minutes * 60))
    if timestamp >= current_bucket:
        raise ValueError("The current unfinished candle cannot be used for a paper decision.")
    fee_rate, slip = fee_bps / 10_000, slippage_bps / 10_000
    with connect(db_path) as con:
        con.execute("BEGIN IMMEDIATE")
        state = con.execute("SELECT cash, asset FROM paper_state WHERE id=1").fetchone()
        cash, asset = (initial_cash, 0.0) if state is None else map(float, state)
        existing = _existing_decision(con, pair, interval_minutes, timestamp)
        if existing is not None:
            return PaperDecision(
                existing[0], float(existing[1]), float(existing[2]), cash, asset,
                float(existing[3]), bool(existing[4]), str(existing[5]), True,
            )
        now = int(time.time())
        if state is None:
            con.execute("INSERT INTO paper_state VALUES (1, ?, ?, ?)", (cash, asset, now))
        equity_before = cash + asset * price
        day_start = timestamp - timestamp % 86400
        day_row = con.execute(
            """SELECT equity FROM paper_equity_snapshots
               WHERE timestamp>=? ORDER BY timestamp LIMIT 1""",
            (day_start,),
        ).fetchone()
        day_equity = float(day_row[0]) if day_row else equity_before
        peak_row = con.execute("SELECT MAX(equity) FROM paper_equity_snapshots").fetchone()
        peak_equity = max(initial_cash, equity_before, float(peak_row[0] or 0))
        daily_return = equity_before / day_equity - 1 if day_equity else 0.0
        drawdown = equity_before / peak_equity - 1 if peak_equity else 0.0
        locked = daily_return <= -daily_loss_limit or drawdown <= -max_drawdown_limit
        action, reason = "hold", "no threshold crossed"
        if locked:
            reason = "daily loss lockout" if daily_return <= -daily_loss_limit else "max drawdown kill switch"
            if asset > 0:
                fill, quantity = price * (1 - slip), asset
                gross, fee = quantity * fill, quantity * fill * fee_rate
                cash, asset, action = cash + gross - fee, 0.0, "sell"
                reason = f"{reason}; risk liquidation"
                con.execute(
                    """INSERT INTO paper_trades
                       (timestamp,side,price,quantity,fee,probability,reason) VALUES(?,?,?,?,?,?,?)""",
                    (timestamp, action, fill, quantity, fee, probability, reason),
                )
        elif probability >= buy_threshold and asset * price < equity_before * max_allocation:
            budget = min(cash, equity_before * max_allocation - asset * price)
            if budget > 1:
                fill, fee = price * (1 + slip), budget * fee_rate
                quantity = max(0.0, (budget - fee) / fill)
                cash, asset, action = cash - budget, asset + quantity, "buy"
                reason = "model probability crossed buy threshold"
                con.execute(
                    """INSERT INTO paper_trades
                       (timestamp,side,price,quantity,fee,probability,reason) VALUES(?,?,?,?,?,?,?)""",
                    (timestamp, action, fill, quantity, fee, probability, reason),
                )
        elif probability <= sell_threshold and asset > 0:
            fill, quantity = price * (1 - slip), asset
            gross, fee = quantity * fill, quantity * fill * fee_rate
            cash, asset, action = cash + gross - fee, 0.0, "sell"
            reason = "model probability crossed sell threshold"
            con.execute(
                """INSERT INTO paper_trades
                   (timestamp,side,price,quantity,fee,probability,reason) VALUES(?,?,?,?,?,?,?)""",
                (timestamp, action, fill, quantity, fee, probability, reason),
            )
        equity = cash + asset * price
        drawdown, daily_return = equity / peak_equity - 1, equity / day_equity - 1
        con.execute(
            """INSERT INTO paper_decisions
               (pair,interval_minutes,timestamp,action,probability,price,equity,locked,reason)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (pair, interval_minutes, timestamp, action, probability, price, equity, int(locked), reason),
        )
        con.execute(
            """INSERT INTO paper_equity_snapshots VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(timestamp) DO NOTHING""",
            (timestamp, equity, cash, asset, drawdown, daily_return),
        )
        con.execute(
            "UPDATE paper_state SET cash=?, asset=?, updated_at=? WHERE id=1",
            (cash, asset, now),
        )
    return PaperDecision(action, probability, price, cash, asset, equity, locked, reason)


def paper_status(db_path: Path, initial_cash: float) -> dict[str, float | int | str | bool]:
    with connect(db_path) as con:
        state = con.execute("SELECT cash, asset, updated_at FROM paper_state WHERE id=1").fetchone()
        latest = con.execute(
            """SELECT timestamp, equity, drawdown, daily_return
               FROM paper_equity_snapshots ORDER BY timestamp DESC LIMIT 1"""
        ).fetchone()
        decisions = int(con.execute("SELECT COUNT(*) FROM paper_decisions").fetchone()[0])
        trades = int(con.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0])
    if state is None:
        return {"cash": initial_cash, "asset": 0.0, "equity": initial_cash, "drawdown": 0.0,
                "daily_return": 0.0, "decisions": 0, "trades": 0, "updated_at": 0}
    return {
        "cash": float(state[0]), "asset": float(state[1]),
        "equity": float(latest[1]) if latest else float(state[0]),
        "drawdown": float(latest[2]) if latest else 0.0,
        "daily_return": float(latest[3]) if latest else 0.0,
        "decisions": decisions, "trades": trades, "updated_at": int(state[2]),
    }


def reset_paper_account(db_path: Path) -> None:
    with connect(db_path) as con:
        con.execute("DELETE FROM paper_trades")
        con.execute("DELETE FROM paper_decisions")
        con.execute("DELETE FROM paper_equity_snapshots")
        con.execute("DELETE FROM paper_state")

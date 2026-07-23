from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .model import positive_probability


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]
    benchmark: pd.DataFrame

    @property
    def total_return(self) -> float:
        return self.metrics["net_return"]

    @property
    def buy_hold_return(self) -> float:
        return self.metrics["buy_hold_return"]

    @property
    def max_drawdown(self) -> float:
        return self.metrics["max_drawdown"]


def _periods_per_year(timestamps: pd.Series) -> float:
    if len(timestamps) < 2:
        return 365.25 * 24
    seconds = float(np.median(np.diff(timestamps.astype(float))))
    return 365.25 * 24 * 3600 / max(seconds, 1)


def _risk_metrics(
    equity: pd.Series, timestamps: pd.Series, initial_equity: float | None = None
) -> dict[str, float]:
    returns = equity.pct_change().dropna()
    periods = _periods_per_year(timestamps)
    years = max((float(timestamps.iloc[-1]) - float(timestamps.iloc[0])) / (365.25 * 86400), 1 / periods)
    starting_equity = float(equity.iloc[0]) if initial_equity is None else initial_equity
    net_return = float(equity.iloc[-1] / starting_equity - 1)
    annualized = float((equity.iloc[-1] / starting_equity) ** (1 / years) - 1)
    std = float(returns.std(ddof=0))
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2))) if len(returns) else 0.0
    sharpe = float(returns.mean() / std * np.sqrt(periods)) if std > 0 else 0.0
    sortino = float(returns.mean() / downside * np.sqrt(periods)) if downside > 0 else 0.0
    drawdown = equity / equity.cummax().clip(lower=starting_equity) - 1
    return {
        "net_return": net_return,
        "annualized_return": annualized,
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "sortino": sortino,
    }


def _trade_statistics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"trade_count": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "turnover": 0.0}
    pnl, entry_cost = [], 0.0
    for row in trades.itertuples(index=False):
        if row.side == "buy":
            entry_cost += row.price * row.quantity + row.fee
        elif entry_cost:
            proceeds = row.price * row.quantity - row.fee
            pnl.append(proceeds - entry_cost)
            entry_cost = 0.0
    wins = [value for value in pnl if value > 0]
    losses = [-value for value in pnl if value < 0]
    profit_factor = sum(wins) / sum(losses) if losses else (float("inf") if wins else 0.0)
    notional = float((trades["price"] * trades["quantity"]).sum())
    return {
        "trade_count": float(len(trades)),
        "win_rate": float(len(wins) / len(pnl)) if pnl else 0.0,
        "profit_factor": float(profit_factor),
        "turnover": notional / float(equity["equity"].mean()),
    }


def run_backtest(
    df: pd.DataFrame,
    model=None,
    initial_cash: float = 1000.0,
    fee_bps: float = 80.0,
    slippage_bps: float = 5.0,
    buy_threshold: float = 0.58,
    sell_threshold: float = 0.48,
    max_allocation: float = 0.25,
    probabilities: np.ndarray | pd.Series | None = None,
) -> BacktestResult:
    if df.empty:
        raise ValueError("Backtest data cannot be empty.")
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    if probabilities is None:
        if model is None:
            if "probability" not in data:
                raise ValueError("Provide a model, probabilities, or a probability column.")
        else:
            data["probability"] = positive_probability(model, data[FEATURE_COLUMNS])
    else:
        data["probability"] = np.asarray(probabilities)
    cash, asset = initial_cash, 0.0
    fee_rate, slip = fee_bps / 10_000, slippage_bps / 10_000
    trades, curve = [], []
    for row in data.itertuples(index=False):
        price, prob = float(row.close), float(row.probability)
        equity_before = cash + asset * price
        if prob >= buy_threshold and asset * price < equity_before * max_allocation:
            budget = min(cash, equity_before * max_allocation - asset * price)
            if budget > 1:
                fill, fee = price * (1 + slip), budget * fee_rate
                qty = max(0.0, (budget - fee) / fill)
                cash, asset = cash - budget, asset + qty
                trades.append((int(row.timestamp), "buy", fill, qty, fee, prob))
        elif prob <= sell_threshold and asset > 0:
            fill, qty = price * (1 - slip), asset
            gross, fee = qty * fill, qty * fill * fee_rate
            cash, asset = cash + gross - fee, 0.0
            trades.append((int(row.timestamp), "sell", fill, qty, fee, prob))
        equity_value = cash + asset * price
        curve.append((int(row.timestamp), equity_value, cash, asset, prob, asset * price / equity_value))
    equity = pd.DataFrame(
        curve, columns=["timestamp", "equity", "cash", "asset", "probability", "exposure"]
    )
    trade_df = pd.DataFrame(
        trades, columns=["timestamp", "side", "price", "quantity", "fee", "probability"]
    )
    metrics = _risk_metrics(equity["equity"], equity["timestamp"], initial_cash)
    metrics.update(_trade_statistics(trade_df, equity))
    metrics["exposure"] = float(equity["exposure"].mean())
    benchmark_equity = initial_cash * data["close"] / float(data["close"].iloc[0])
    benchmark = pd.DataFrame({"timestamp": data["timestamp"], "equity": benchmark_equity})
    benchmark_metrics = _risk_metrics(benchmark["equity"], benchmark["timestamp"], initial_cash)
    metrics.update({f"buy_hold_{key}": value for key, value in benchmark_metrics.items()})
    metrics["buy_hold_return"] = benchmark_metrics["net_return"]
    return BacktestResult(equity, trade_df, metrics, benchmark)


def sensitivity_table(
    predictions: pd.DataFrame,
    initial_cash: float,
    fee_values: tuple[float, ...],
    slippage_values: tuple[float, ...],
    buy_threshold: float,
    sell_threshold: float,
    max_allocation: float,
) -> pd.DataFrame:
    rows = []
    for fee in fee_values:
        for slippage in slippage_values:
            result = run_backtest(
                predictions,
                initial_cash=initial_cash,
                fee_bps=fee,
                slippage_bps=slippage,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                max_allocation=max_allocation,
            )
            rows.append({"fee_bps": fee, "slippage_bps": slippage, **result.metrics})
    return pd.DataFrame(rows)

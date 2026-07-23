import asyncio
import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .backtest import run_backtest, sensitivity_table
from .config import settings
from .db import connect, load_candles, upsert_candles
from .experiment import load_experiment_config, run_walk_forward
from .features import build_features
from .history import HistorySummary, sync_history
from .kraken import KrakenPublicClient
from .model import load_model, train_chronological
from .paper import execute_paper_step, paper_status, reset_paper_account

app = typer.Typer(no_args_is_help=True)
console = Console()


def _features(include_targets: bool = True) -> pd.DataFrame:
    candles = load_candles(settings.db_path, settings.pair, settings.interval_minutes)
    return build_features(
        candles,
        fee_bps=settings.taker_fee_bps,
        slippage_bps=settings.slippage_bps,
        include_targets=include_targets,
    )


@app.command()
def fetch() -> None:
    """Fetch Kraken's recent completed OHLC candles into SQLite."""
    client = KrakenPublicClient()
    try:
        df = client.fetch_ohlc(settings.rest_pair, settings.interval_minutes)
    finally:
        client.close()
    changes = upsert_candles(settings.db_path, df, settings.pair, settings.interval_minutes)
    console.print(f"Stored/updated {changes} candles; received {len(df)} completed candles.")


@app.command("sync-history")
def sync_history_command() -> None:
    """Download, resume, reconcile, and verify all available BTC/EUR 1h history."""
    def show_progress(batch: int, inserted: int, cursor: str) -> None:
        if batch == 1 or batch % 100 == 0:
            console.print(
                f"History batch {batch}: {inserted} new trades; checkpoint {cursor}",
                highlight=False,
            )

    summary = sync_history(
        settings.db_path,
        settings.pair,
        settings.rest_pair,
        settings.interval_minutes,
        request_delay=settings.history_request_delay,
        progress=show_progress,
    )
    console.print(f"First candle: {HistorySummary.format_timestamp(summary.first_candle)}")
    console.print(f"Last candle: {HistorySummary.format_timestamp(summary.last_candle)}")
    console.print(f"Number of candles: {summary.candles:,}")
    console.print(f"Missing candles: {summary.missing_candles:,}")
    console.print(f"Database size: {summary.database_size:,} bytes")


@app.command()
def train() -> None:
    """Train the calibrated chronological baseline and save it for paper inference."""
    result = train_chronological(_features(), settings.model_path)
    auc = "n/a" if result.test_auc is None else f"{result.test_auc:.3f}"
    console.print(
        f"Model saved to {settings.model_path}; train={result.train_rows}, "
        f"test={result.test_rows}, AUC={auc}"
    )


def _write_backtest_reports(predictions: pd.DataFrame, config: dict | None = None) -> None:
    backtest_config = {} if config is None else config.get("backtest", {})
    initial_cash = float(backtest_config.get("initial_cash", settings.initial_cash))
    fee_bps = float(backtest_config.get("fee_bps_per_fill", settings.taker_fee_bps))
    slippage_bps = float(
        backtest_config.get("slippage_bps_per_fill", settings.slippage_bps)
    )
    buy_threshold = float(backtest_config.get("buy_threshold", settings.buy_threshold))
    sell_threshold = float(backtest_config.get("sell_threshold", settings.sell_threshold))
    max_allocation = float(backtest_config.get("max_allocation", settings.max_allocation))
    result = run_backtest(
        predictions,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        max_allocation=max_allocation,
    )
    sensitivity = sensitivity_table(
        predictions,
        initial_cash,
        tuple(backtest_config.get("sensitivity_fee_bps", (40.0, 80.0, 120.0))),
        tuple(backtest_config.get("sensitivity_slippage_bps", (0.0, 5.0, 10.0))),
        buy_threshold,
        sell_threshold,
        max_allocation,
    )
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    result.equity.to_csv(report_dir / "equity.csv", index=False)
    result.trades.to_csv(report_dir / "trades.csv", index=False)
    result.benchmark.to_csv(report_dir / "benchmark.csv", index=False)
    sensitivity.to_csv(report_dir / "sensitivity.csv", index=False)
    pd.DataFrame([result.metrics]).to_csv(report_dir / "metrics.csv", index=False)
    table = Table("Metric", "Value")
    for key in (
        "net_return", "annualized_return", "max_drawdown", "sharpe", "sortino",
        "trade_count", "win_rate", "profit_factor", "exposure", "turnover", "buy_hold_return",
    ):
        value = result.metrics[key]
        table.add_row(key.replace("_", " ").title(), f"{value:.4f}")
    console.print(table)


@app.command("walk-forward")
def walk_forward(config: Path = typer.Option(settings.experiment_config, "--config")) -> None:
    """Run and persist expanding-window out-of-sample folds."""
    experiment_config = load_experiment_config(config)
    features = build_features(
        load_candles(settings.db_path, settings.pair, settings.interval_minutes),
        horizon=int(experiment_config["horizon"]),
        fee_bps=float(experiment_config["label"]["fee_bps_per_fill"]),
        slippage_bps=float(experiment_config["label"]["slippage_bps_per_fill"]),
    )
    result = run_walk_forward(features, experiment_config, settings.db_path, settings.model_path)
    Path("reports").mkdir(exist_ok=True)
    result.predictions.to_csv("reports/oos_predictions.csv", index=False)
    result.folds.to_csv("reports/folds.csv", index=False)
    _write_backtest_reports(result.predictions, experiment_config)
    console.print(
        f"Experiment {result.experiment_id}: {len(result.folds)} folds, "
        f"{len(result.predictions)} strictly out-of-sample predictions."
    )


@app.command()
def backtest(experiment_id: str | None = typer.Option(None, "--experiment-id")) -> None:
    """Rebuild reports from persisted out-of-sample predictions."""
    with connect(settings.db_path) as con:
        if experiment_id is None:
            row = con.execute(
                """SELECT id, config_json FROM experiments WHERE status='complete'
                   ORDER BY created_at DESC LIMIT 1"""
            ).fetchone()
            if row is None:
                raise typer.BadParameter("No experiment found; run 'olsen walk-forward' first.")
            experiment_id = str(row[0])
            experiment_config = json.loads(row[1])
        else:
            row = con.execute(
                "SELECT config_json FROM experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise typer.BadParameter(f"Experiment not found: {experiment_id}")
            experiment_config = json.loads(row[0])
        predictions = pd.read_sql_query(
            """SELECT timestamp, close, probability FROM experiment_predictions
               WHERE experiment_id=? ORDER BY timestamp""",
            con,
            params=(experiment_id,),
        )
    if predictions.empty:
        raise typer.BadParameter(f"No predictions found for experiment {experiment_id}")
    _write_backtest_reports(predictions, experiment_config)
    console.print(f"Reports reproduced from experiment {experiment_id}.")


@app.command()
def paper() -> None:
    """Fetch newest candles and record one idempotent paper decision."""
    fetch()
    features = _features(include_targets=False)
    model = load_model(settings.model_path)
    decision = execute_paper_step(
        settings.db_path, features.tail(1), model, settings.initial_cash,
        settings.taker_fee_bps, settings.slippage_bps, settings.buy_threshold,
        settings.sell_threshold, settings.max_allocation, settings.pair,
        settings.interval_minutes, settings.daily_loss_limit, settings.max_drawdown_limit,
    )
    suffix = " | already recorded" if decision.duplicate else ""
    console.print(
        f"{decision.action.upper()} | p={decision.probability:.3f} | price={decision.price:.2f} | "
        f"equity={decision.equity:.2f} | cash={decision.cash:.2f} | "
        f"asset={decision.asset:.8f} | {decision.reason}{suffix}"
    )


@app.command("paper-status")
def show_paper_status() -> None:
    """Show paper balances, risk state, decisions and trades."""
    state = paper_status(settings.db_path, settings.initial_cash)
    table = Table("Field", "Value")
    for key, value in state.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


@app.command("paper-reset")
def paper_reset(confirm: bool = typer.Option(False, "--confirm")) -> None:
    """Reset only the paper account; requires an explicit --confirm flag."""
    if not confirm:
        raise typer.BadParameter("Refusing to reset without --confirm")
    reset_paper_account(settings.db_path)
    console.print("Paper account state, decisions, snapshots and trades reset.")


@app.command("collect-ws")
def collect_ws() -> None:
    """Run the optional Kraken WebSocket v2 OHLC collector with REST reconciliation."""
    from .websocket import collect_ohlc

    asyncio.run(collect_ohlc(
        settings.db_path, settings.pair, settings.interval_minutes, settings.rest_pair
    ))


@app.command()
def dashboard(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the optional read-only dashboard API."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install Olsen with the 'services' extra.") from exc
    from .dashboard import create_app

    uvicorn.run(create_app(settings.db_path), host=host, port=port)


@app.command()
def status() -> None:
    candles = load_candles(settings.db_path, settings.pair, settings.interval_minutes)
    console.print(
        f"Pair: {settings.pair}; interval: {settings.interval_minutes}m; candles: {len(candles)}; "
        "live execution: disabled"
    )
    if len(candles):
        console.print(f"Latest close: {candles.iloc[-1]['close']:.2f}")


if __name__ == "__main__":
    app()

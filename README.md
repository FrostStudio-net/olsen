# Olsen

Olsen is a Kraken-first machine-learning research and paper-trading system for hourly
BTC/EUR (`XBT/EUR`). Version 0.2 cannot submit live orders. It is research software,
not evidence of profitability and not financial advice.

## Safety boundary

- Spot paper trading only; long or cash.
- Live execution is hard-disabled and the repository has no credential settings.
- The current unfinished Kraken candle is excluded from REST, WebSocket persistence,
  and paper decisions.
- All evaluation is chronological. Forward labels crossing a fold boundary are purged.
- Private withdrawal permissions are neither requested nor stored.
- A deterministic client-order-ID helper exists for future reconciliation work, but
  the private adapter always refuses order placement.

## Install and test

```bash
cd /absolute/path/to/olsen
python3 -m venv .venv
source .venv/bin/activate
pip install --no-build-isolation -e '.[dev]'
cp .env.example .env
pytest
ruff check .
```

Install optional read-only services separately:

```bash
pip install --no-build-isolation -e '.[services]'
```

## Data

Kraken REST supplies recent data:

```bash
olsen fetch
olsen status
```

For meaningful two-year expanding windows, download Kraken's official historical
OHLCVT ZIP, extract the 60-minute BTC/EUR CSV, and run:

```bash
olsen import-csv /absolute/path/to/XBTEUR_60.csv
olsen fetch
```

The expected CSV columns are `timestamp,open,high,low,close,volume,trades`.

## Reproducible research workflow

The committed experiment configuration is `configs/v0.2.json`. It defines the market,
horizon, costs, labels, fold frequency, and model family.

```bash
olsen walk-forward --config configs/v0.2.json
olsen backtest
# Or reproduce one persisted run exactly:
olsen backtest --experiment-id EXPERIMENT_ID
```

Walk-forward validation uses expanding training data, a minimum two-year history when
the available span permits it, quarterly folds by default, a purged forward-label
boundary, fold-local scaling/model fitting, and chronological holdout probability
calibration. It persists configuration, configuration/model hashes, fold metrics, and
each out-of-sample prediction in SQLite.

Reports are written to:

- `reports/oos_predictions.csv` and `reports/folds.csv`
- `reports/equity.csv`, `reports/trades.csv`, and `reports/benchmark.csv`
- `reports/metrics.csv` with return, risk, trade, exposure, and turnover metrics
- `reports/sensitivity.csv` for fee/slippage assumptions

The cost-aware target predicts forward return after estimated round-trip fees and
slippage. Olsen trains a binary baseline and a three-way label: downside risk, no edge,
or buy opportunity. Market-regime inputs cover long-trend slope, historical realized-
volatility percentile, and drawdown from a rolling high, using no future observations.

For a simple final chronological model used only by paper inference:

```bash
olsen train
```

## Paper account

```bash
olsen paper
olsen paper-status
olsen paper-reset --confirm
```

`olsen paper` fetches REST data first and records exactly one decision per completed
candle. A database constraint makes reruns idempotent. Equity snapshots drive the
configured daily-loss lockout and maximum-drawdown kill switch. Reset deletes only
paper state, decisions, snapshots, and trades; it requires the literal `--confirm` flag.

Run shortly after each hourly candle closes:

```cron
5 * * * * cd /absolute/path/to/olsen && .venv/bin/olsen paper >> data/paper.log 2>&1
```

## Optional market-data collector and dashboard

The WebSocket v2 collector retains REST reconciliation and stores completed candles
only:

```bash
olsen collect-ws
```

The dashboard binds locally by default and exposes read-only health, candle,
experiment/prediction, and paper-account endpoints:

```bash
olsen dashboard --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
curl 'http://127.0.0.1:8000/candles?limit=10'
curl http://127.0.0.1:8000/experiments
curl http://127.0.0.1:8000/paper
```

## Before any future live-money work

Live execution must remain absent until leakage reviews, untouched-data evaluation,
forward paper testing, risk-limit validation, a manual kill switch, deterministic order
IDs, and idempotent reconciliation have all been independently verified. Any future API
key must have no withdrawal permission.

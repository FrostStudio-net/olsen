# Codex brief: Olsen v0.2

You are working on Olsen, a Kraken-first ML trading and paper-trading system. Read the entire repository before changing anything.

## Non-negotiable safety constraints

- Keep live order placement disabled by default.
- Never request or store withdrawal permissions.
- Do not claim profitability.
- Do not introduce look-ahead bias.
- Never train on rows later than the test rows.
- The current unfinished Kraken candle must never be used for a decision.
- Every future order must have a deterministic client order ID and reconciliation logic.

## Implement next

1. Add expanding-window walk-forward validation:
   - Minimum 2 years training when data allows.
   - Monthly or quarterly test folds.
   - Fit scaler/model only inside each training fold.
   - Combine all out-of-sample predictions into one report.

2. Add metrics:
   - Net return, annualized return, max drawdown, Sharpe, Sortino.
   - Trade count, win rate, profit factor, exposure and turnover.
   - BTC/EUR buy-and-hold benchmark.
   - Fee and slippage sensitivity table.

3. Improve targets:
   - Predict forward return after estimated round-trip costs.
   - Add a three-way label: buy opportunity, no edge, downside risk.
   - Keep binary model as baseline.

4. Add market-regime features:
   - Trend regime from long moving-average slope.
   - Realized-volatility percentile using past data only.
   - Drawdown from rolling high.
   - Never use future information.

5. Add experiment persistence:
   - SQLite tables for experiment configuration, model hash, fold metrics and predictions.
   - Every report must be reproducible from a committed config.

6. Harden paper trading:
   - Record one decision per candle with a unique database constraint.
   - Add equity snapshots and daily-loss lockout.
   - Add max drawdown kill switch.
   - Add `olsen paper-status` and `olsen paper-reset --confirm`.

7. Add Kraken WebSocket v2 market-data collector as an optional service while retaining REST reconciliation.

8. Add a minimal FastAPI read-only dashboard API only after the research layer is tested.

## Acceptance criteria

- `pytest` passes.
- `ruff check .` passes.
- A synthetic test proves future price changes cannot alter earlier features.
- Re-running the same paper candle cannot create a duplicate trade.
- Walk-forward output contains only predictions made by models trained on earlier timestamps.
- README includes exact commands.

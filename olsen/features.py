import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "volatility_12", "volatility_24", "ema_gap_12_48",
    "rsi_14", "atr_pct_14", "volume_ratio_24", "range_pct",
    "trend_slope_168", "volatility_percentile_720", "drawdown_720",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _past_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile of the current observation against an earlier-only window."""
    return series.rolling(window, min_periods=48).apply(
        lambda values: float(np.mean(values[:-1] <= values[-1])) if len(values) > 1 else np.nan,
        raw=True,
    )


def build_features(
    candles: pd.DataFrame,
    horizon: int = 12,
    fee_bps: float = 80.0,
    slippage_bps: float = 5.0,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Build causal features and, optionally, forward cost-aware labels.

    Feature columns use only the current and earlier completed candles. Targets are
    kept separate from inference so the newest completed candle remains usable.
    """
    df = candles.copy().sort_values("timestamp").reset_index(drop=True)
    close = df["close"]
    returns = close.pct_change()
    for n in (1, 3, 6, 12, 24):
        df[f"ret_{n}"] = close.pct_change(n)
    df["volatility_12"] = returns.rolling(12).std()
    df["volatility_24"] = returns.rolling(24).std()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema48 = close.ewm(span=48, adjust=False).mean()
    df["ema_gap_12_48"] = ema12 / ema48 - 1
    df["rsi_14"] = _rsi(close)
    prev_close = close.shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_pct_14"] = true_range.rolling(14).mean() / close
    df["volume_ratio_24"] = df["volume"] / df["volume"].rolling(24).mean()
    df["range_pct"] = (df["high"] - df["low"]) / close
    long_ma = close.rolling(168, min_periods=48).mean()
    df["trend_slope_168"] = long_ma / long_ma.shift(24) - 1
    realized_vol = returns.rolling(24).std()
    df["volatility_percentile_720"] = _past_percentile(realized_vol, 720)
    df["drawdown_720"] = close / close.rolling(720, min_periods=48).max() - 1
    dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    feature_mask = df[FEATURE_COLUMNS].notna().all(axis=1)
    if not include_targets:
        return df.loc[feature_mask].reset_index(drop=True)

    round_trip_cost = 2 * (fee_bps + slippage_bps) / 10_000
    df["future_return"] = close.shift(-horizon) / close - 1
    df["net_forward_return"] = df["future_return"] - round_trip_cost
    df["target"] = (df["net_forward_return"] > 0).astype(int)
    df["target_3way"] = np.select(
        [df["future_return"] > round_trip_cost, df["future_return"] < -round_trip_cost],
        [1, -1],
        default=0,
    )
    df["target_timestamp"] = df["timestamp"].shift(-horizon)
    target_mask = df[["future_return", "target_timestamp"]].notna().all(axis=1)
    return df.loc[feature_mask & target_mask].reset_index(drop=True)

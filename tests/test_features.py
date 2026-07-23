import numpy as np
import pandas as pd

from olsen.features import FEATURE_COLUMNS, build_features


def test_features_are_complete():
    n = 200
    close = 100 + np.arange(n) * 0.1 + np.sin(np.arange(n) / 4)
    df = pd.DataFrame({
        "timestamp": 1_700_000_000 + np.arange(n) * 3600,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "vwap": close,
        "volume": 10 + np.arange(n) % 7,
        "trades": 100,
    })
    out = build_features(df)
    assert len(out) > 50
    assert not out[FEATURE_COLUMNS].isna().any().any()
    assert set(out["target"].unique()).issubset({0, 1})
    assert set(out["target_3way"].unique()).issubset({-1, 0, 1})


def test_future_price_changes_cannot_alter_earlier_features():
    n = 1000
    close = 100 + np.arange(n) * 0.02 + np.sin(np.arange(n) / 8)
    candles = pd.DataFrame({
        "timestamp": 1_600_000_000 + np.arange(n) * 3600,
        "open": close - 0.1,
        "high": close + 0.4,
        "low": close - 0.4,
        "close": close,
        "vwap": close,
        "volume": 10 + np.arange(n) % 11,
        "trades": 100,
    })
    cutoff = int(candles.iloc[800]["timestamp"])
    original = build_features(candles, include_targets=False)
    changed = candles.copy()
    changed.loc[changed["timestamp"] > cutoff, ["open", "high", "low", "close", "vwap"]] *= 4
    rebuilt = build_features(changed, include_targets=False)
    left = original[original["timestamp"] <= cutoff].set_index("timestamp")[FEATURE_COLUMNS]
    right = rebuilt[rebuilt["timestamp"] <= cutoff].set_index("timestamp")[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(left, right)


def test_inference_keeps_latest_completed_feature_row():
    n = 300
    close = 100 + np.sin(np.arange(n) / 6) + np.arange(n) * 0.01
    candles = pd.DataFrame({
        "timestamp": 1_600_000_000 + np.arange(n) * 3600,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "vwap": close,
        "volume": 10 + np.arange(n) % 5,
        "trades": 10,
    })
    inference = build_features(candles, include_targets=False)
    assert inference.iloc[-1]["timestamp"] == candles.iloc[-1]["timestamp"]

import numpy as np
import pandas as pd

from olsen.experiment import run_walk_forward
from olsen.features import build_features


def test_walk_forward_models_train_only_before_test_rows(tmp_path):
    n = 1000
    index = np.arange(n)
    close = 100 + index * 0.03 + 4 * np.sin(index / 9) + 2 * np.sin(index / 31)
    candles = pd.DataFrame({
        "timestamp": 1_500_000_000 + index * 86400,
        "open": close - 0.2,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "vwap": close,
        "volume": 10 + index % 17,
        "trades": 50 + index % 13,
    })
    features = build_features(candles, horizon=3, fee_bps=10, slippage_bps=2)
    config = {
        "name": "synthetic",
        "pair": "XBT/EUR",
        "interval_minutes": 1440,
        "horizon": 3,
        "fold_frequency": "quarterly",
        "label": {"fee_bps_per_fill": 10, "slippage_bps_per_fill": 2},
        "model": {"type": "hist_gradient_boosting", "calibration": "chronological_holdout"},
    }

    result = run_walk_forward(features, config, tmp_path / "experiments.db")

    assert len(result.predictions) > 0
    assert (result.predictions["train_end"] < result.predictions["timestamp"]).all()
    for fold in result.folds.itertuples(index=False):
        assert fold.train_end < fold.test_start
        assert fold.test_start - fold.train_start >= 730 * 86400

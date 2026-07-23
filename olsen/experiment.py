import hashlib
import json
import math
import pickle
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .backtest import run_backtest
from .db import connect
from .features import FEATURE_COLUMNS
from .model import fit_model, positive_probability


@dataclass
class WalkForwardResult:
    experiment_id: str
    predictions: pd.DataFrame
    folds: pd.DataFrame
    model_hash: str


def load_experiment_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"name", "pair", "interval_minutes", "horizon", "fold_frequency", "model"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Experiment config is missing: {', '.join(sorted(missing))}")
    return config


def _canonical_json(config: dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _hash_model(model) -> str:
    return hashlib.sha256(pickle.dumps(model)).hexdigest()


def _test_periods(df: pd.DataFrame, frequency: str) -> tuple[list[pd.Period], int]:
    aliases = {"monthly": "M", "quarterly": "Q"}
    if frequency not in aliases:
        raise ValueError("fold_frequency must be 'monthly' or 'quarterly'")
    dates = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    span = int(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])
    periods = dates.dt.to_period(aliases[frequency])
    if span >= 2 * 365 * 86400:
        earliest = pd.to_datetime(int(df["timestamp"].iloc[0]), unit="s") + pd.Timedelta(days=730)
        eligible = periods.map(lambda period: period.start_time >= earliest)
        minimum_timestamp = int(earliest.timestamp())
    else:
        eligible = pd.Series(False, index=df.index)
        cutoff = max(320, int(len(df) * 0.6))
        if cutoff < len(df):
            eligible.iloc[cutoff:] = True
            minimum_timestamp = int(df.iloc[cutoff]["timestamp"])
        else:
            minimum_timestamp = int(df.iloc[-1]["timestamp"]) + 1
    return list(periods[eligible].drop_duplicates()), minimum_timestamp


def run_walk_forward(
    df: pd.DataFrame,
    config: dict,
    db_path: Path,
    model_path: Path | None = None,
) -> WalkForwardResult:
    """Fit expanding, purged temporal folds and persist every OOS prediction."""
    data = df.sort_values("timestamp").reset_index(drop=True)
    periods, minimum_test_timestamp = _test_periods(data, config["fold_frequency"])
    if not periods:
        raise ValueError("Not enough rows for a walk-forward test fold.")
    created_at = time.time_ns()
    config_json = _canonical_json(config)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    experiment_id = f"{config_hash[:12]}-{created_at}-{uuid.uuid4().hex[:8]}"
    prediction_frames, fold_rows, fold_models = [], [], []
    dates = pd.to_datetime(data["timestamp"], unit="s", utc=True).dt.tz_localize(None)
    period_alias = "M" if config["fold_frequency"] == "monthly" else "Q"
    for fold_id, period in enumerate(periods, start=1):
        test_mask = dates.dt.to_period(period_alias) == period
        test = data.loc[test_mask & (data["timestamp"] >= minimum_test_timestamp)]
        if test.empty:
            continue
        test_start = int(test["timestamp"].min())
        train = data[data["target_timestamp"] < test_start]
        if len(train) < 300 or train["target"].nunique() < 2 or train["target_3way"].nunique() < 2:
            continue
        binary_model = fit_model(train, "target", calibrate=True)
        three_way_model = fit_model(train, "target_3way", calibrate=False)
        probability = positive_probability(binary_model, test[FEATURE_COLUMNS])
        probability_buy = positive_probability(three_way_model, test[FEATURE_COLUMNS])
        three_way_prediction = three_way_model.classes_[
            three_way_model.predict_proba(test[FEATURE_COLUMNS]).argmax(axis=1)
        ]
        auc = roc_auc_score(test["target"], probability) if test["target"].nunique() > 1 else None
        backtest_config = config.get("backtest", {})
        fold_backtest = run_backtest(
            test,
            initial_cash=float(backtest_config.get("initial_cash", 1000.0)),
            fee_bps=float(backtest_config.get("fee_bps_per_fill", 80.0)),
            slippage_bps=float(backtest_config.get("slippage_bps_per_fill", 5.0)),
            buy_threshold=float(backtest_config.get("buy_threshold", 0.58)),
            sell_threshold=float(backtest_config.get("sell_threshold", 0.48)),
            max_allocation=float(backtest_config.get("max_allocation", 0.25)),
            probabilities=probability,
        )
        metrics = {
            "auc": auc,
            "three_way_accuracy": float(accuracy_score(test["target_3way"], three_way_prediction)),
            **fold_backtest.metrics,
        }
        persisted_metrics = {
            key: value if not isinstance(value, float) or math.isfinite(value) else None
            for key, value in metrics.items()
        }
        model_hash = _hash_model((binary_model, three_way_model))
        fold_models.append(model_hash)
        fold_rows.append({
            "fold_id": fold_id,
            "train_start": int(train["timestamp"].min()),
            "train_end": int(train["timestamp"].max()),
            "test_start": test_start,
            "test_end": int(test["timestamp"].max()),
            "train_rows": len(train),
            "test_rows": len(test),
            "model_hash": model_hash,
            "metrics_json": json.dumps(persisted_metrics, sort_keys=True, allow_nan=False),
        })
        frame = test[["timestamp", "close", "target", "target_3way"]].copy()
        frame["fold_id"] = fold_id
        frame["probability"] = probability
        frame["probability_buy_3way"] = probability_buy
        frame["train_end"] = int(train["timestamp"].max())
        if not (frame["train_end"] < frame["timestamp"]).all():
            raise AssertionError("Walk-forward leakage: training reaches a test timestamp")
        prediction_frames.append(frame)
    if not prediction_frames:
        raise ValueError("No valid walk-forward folds; import more diverse history.")
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values("timestamp")
    folds = pd.DataFrame(fold_rows)
    combined_hash = hashlib.sha256("".join(fold_models).encode()).hexdigest()
    with connect(db_path) as con:
        con.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?)",
            (experiment_id, created_at, config_json, config_hash, combined_hash, "complete"),
        )
        con.executemany(
            """INSERT INTO experiment_folds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (experiment_id, int(r.fold_id), int(r.train_start), int(r.train_end),
                 int(r.test_start), int(r.test_end), int(r.train_rows), int(r.test_rows),
                 r.model_hash, r.metrics_json)
                for r in folds.itertuples(index=False)
            ],
        )
        con.executemany(
            """INSERT INTO experiment_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (experiment_id, int(r.fold_id), int(r.timestamp), float(r.close), int(r.target),
                 int(r.target_3way), float(r.probability), float(r.probability_buy_3way),
                 int(r.train_end))
                for r in predictions.itertuples(index=False)
            ],
        )
    if model_path is not None:
        final_model = fit_model(data, "target", calibrate=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": final_model, "features": FEATURE_COLUMNS, "version": "0.2.0",
             "config_hash": config_hash},
            model_path,
        )
    return WalkForwardResult(experiment_id, predictions, folds, combined_hash)

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass
class TrainingResult:
    model: object
    train_rows: int
    test_rows: int
    test_auc: float | None


class CalibratedModel:
    """Temporal holdout calibration without non-chronological cross-validation."""

    def __init__(self, estimator: Pipeline, calibrator: LogisticRegression | None = None) -> None:
        self.estimator = estimator
        self.calibrator = calibrator

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(x)
        if self.calibrator is None or raw.shape[1] != 2:
            return raw
        calibrated = self.calibrator.predict_proba(raw[:, 1].reshape(-1, 1))[:, 1]
        return np.column_stack([1 - calibrated, calibrated])


def _pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])


def fit_model(train: pd.DataFrame, target_column: str = "target", calibrate: bool = True):
    if train[target_column].nunique() < 2:
        raise ValueError("Training data must contain at least two target classes.")
    calibration_rows = max(60, len(train) // 5)
    use_calibration = calibrate and target_column == "target" and len(train) >= 300
    base = train.iloc[:-calibration_rows] if use_calibration else train
    if use_calibration and base[target_column].nunique() < 2:
        use_calibration = False
        base = train
    estimator = _pipeline().fit(base[FEATURE_COLUMNS], base[target_column])
    calibrator = None
    if use_calibration:
        calibration = train.iloc[-calibration_rows:]
        raw = estimator.predict_proba(calibration[FEATURE_COLUMNS])[:, 1]
        if calibration[target_column].nunique() > 1:
            calibrator = LogisticRegression(random_state=42).fit(
                raw.reshape(-1, 1), calibration[target_column]
            )
    return CalibratedModel(estimator, calibrator)


def positive_probability(model, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x))
    return probabilities[:, classes.index(1)]


def train_chronological(df: pd.DataFrame, model_path: Path, test_fraction: float = 0.2) -> TrainingResult:
    if len(df) < 300:
        raise ValueError("Need at least 300 feature rows; import more history before training.")
    split = int(len(df) * (1 - test_fraction))
    test = df.iloc[split:]
    test_start = int(test["timestamp"].min())
    train = df.iloc[:split]
    if "target_timestamp" in train:
        train = train[train["target_timestamp"] < test_start]
    model = fit_model(train)
    prob = positive_probability(model, test[FEATURE_COLUMNS])
    auc = roc_auc_score(test["target"], prob) if test["target"].nunique() > 1 else None
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "version": "0.2.0"}, model_path)
    return TrainingResult(model, len(train), len(test), auc)


def load_model(path: Path):
    payload = joblib.load(path)
    return payload["model"]

import sqlite3

import numpy as np
import pandas as pd

from olsen.features import FEATURE_COLUMNS
from olsen.paper import execute_paper_step


class BuyModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, values):
        return np.tile([0.1, 0.9], (len(values), 1))


def test_same_paper_candle_cannot_create_duplicate_trade(tmp_path):
    row = {column: 0.0 for column in FEATURE_COLUMNS}
    row.update({"timestamp": 1_700_000_000, "close": 30_000.0})
    feature_row = pd.DataFrame([row])
    db_path = tmp_path / "paper.db"
    args = (db_path, feature_row, BuyModel(), 1000.0, 80.0, 5.0, 0.58, 0.48, 0.25)

    first = execute_paper_step(*args)
    second = execute_paper_step(*args)

    assert first.action == "buy"
    assert not first.duplicate
    assert second.duplicate
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM paper_decisions").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 1

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="OLSEN_", extra="ignore")

    pair: str = "XBT/EUR"
    rest_pair: str = "XBTEUR"
    interval_minutes: int = 60
    initial_cash: float = 1000.0
    taker_fee_bps: float = 80.0
    slippage_bps: float = 5.0
    buy_threshold: float = 0.58
    sell_threshold: float = 0.48
    max_allocation: float = 0.25
    daily_loss_limit: float = 0.03
    max_drawdown_limit: float = 0.15
    walk_forward_frequency: str = "quarterly"
    experiment_config: Path = Path("configs/v0.2.json")
    db_path: Path = Path("data/olsen.db")
    model_path: Path = Path("models/model.joblib")


settings = Settings()

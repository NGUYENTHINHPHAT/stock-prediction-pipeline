"""
End-to-end pipeline steps shared by the CLI scripts and the API.

    fetch (Yahoo Finance) -> validate & clean -> engineer features -> train -> save
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data.features import FeatureEngineer
from src.data.fetcher import DataFetcher
from src.data.validator import DataValidator
from src.models.trainer import ModelTrainer
from src.models.xgboost_model import XGBoostModel
from src.utils.config import settings
from src.utils.logger import log

MODEL_DIR = Path("models/saved")
LATEST_MODEL = MODEL_DIR / "latest_model.pkl"
PARAMS_FILE = Path("models/best_params.json")
RESULTS_DIR = Path("results")


def build_dataset(
    symbols: Optional[List[str]] = None,
    lookback_days: Optional[int] = None,
    horizon: Optional[int] = None,
) -> pd.DataFrame:
    """Download OHLCV history and return the feature matrix with targets."""
    symbols = symbols or settings.symbol_list
    lookback_days = lookback_days or settings.lookback_days
    horizon = horizon or settings.prediction_horizon

    df = DataFetcher().fetch(symbols, start_date=datetime.now() - timedelta(days=lookback_days))

    validator = DataValidator()
    validator.validate(df)
    df = validator.clean_data(df)

    df = FeatureEngineer(window=settings.feature_window).engineer_features(
        df, create_target=True, target_horizon=horizon
    )
    log.info(f"Dataset ready: {len(df)} rows x {df.shape[1]} cols, "
             f"{df['date'].min().date()} -> {df['date'].max().date()}, horizon={horizon}d")
    return df


def load_params(path: Optional[Path] = PARAMS_FILE) -> Optional[Dict]:
    """Tuned XGBoost params from JSON, or None to use the model defaults."""
    if path is None or not Path(path).exists():
        log.info("No tuned params file - using XGBoost defaults")
        return None
    text = Path(path).read_text().strip()
    if not text:
        return None
    log.info(f"Loaded params from {path}")
    return json.loads(text)


def train_and_save(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
    model_name: str = "xgboost",
    mlflow_tracking: bool = True,
) -> Tuple[XGBoostModel, Dict[str, float]]:
    """Fit on a purged chronological train/val/test split, then save the model."""
    model = XGBoostModel(model_name=model_name, params=params)
    metrics = ModelTrainer(mlflow_tracking=mlflow_tracking).train_model(
        model=model,
        df=df,
        target_col="target_direction",
        train_size=0.7,
        val_size=0.15,
        log_mlflow=mlflow_tracking,
        early_stopping_rounds=10,
        verbose=False,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(str(MODEL_DIR / f"{model_name}_{stamp}.pkl"))
    model.save(str(LATEST_MODEL))
    log.info(f"Model saved -> {LATEST_MODEL}")

    importance = model.get_feature_importance()
    if importance is not None:
        RESULTS_DIR.mkdir(exist_ok=True)
        importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    return model, metrics

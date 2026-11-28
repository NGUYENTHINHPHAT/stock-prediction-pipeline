"""
Configuration management using Pydantic Settings
"""
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Model Configuration
    prediction_horizon: int = Field(
        default=5,
        description="Days ahead to predict"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    
    # Data Configuration
    symbols: str = Field(
        default="AAPL,MSFT,GOOGL,AMZN,TSLA",
        description="Comma-separated stock symbols"
    )
    lookback_days: int = Field(
        default=1825,
        description="Days of historical data to fetch"
    )
    feature_window: int = Field(
        default=20,
        description="Window for technical indicators"
    )
    
    # MLflow
    mlflow_tracking_uri: str = Field(
        default="./mlruns",
        description="MLflow tracking URI"
    )
    mlflow_experiment_name: str = Field(
        default="stock_prediction",
        description="MLflow experiment name"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="./logs/app.log", description="Log file path")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": (),
        "extra": "ignore",
    }
    
    @property
    def symbol_list(self) -> List[str]:
        """Parse symbols into list"""
        return [s.strip() for s in self.symbols.split(",")]


# Global settings instance
settings = Settings()

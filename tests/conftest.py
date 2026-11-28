"""Shared fixtures: synthetic market data so the suite runs offline and fast."""
import numpy as np
import pandas as pd
import pytest

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def make_ohlcv(n_days: int = 260, symbols=SYMBOLS, seed: int = 0) -> pd.DataFrame:
    """Geometric random-walk OHLCV panel, one row per (business day, symbol)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    frames = []
    for sym in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n_days)))
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        frames.append(pd.DataFrame({
            "date": dates,
            "symbol": sym,
            "open": open_,
            "high": np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n_days)),
            "low": np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n_days)),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def feature_panel(ohlcv) -> pd.DataFrame:
    """Minimal model-ready frame: two noise features plus a 5-day target."""
    rng = np.random.default_rng(1)
    df = ohlcv.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["future_close"] = df.groupby("symbol")["close"].shift(-5)
    df["target_return"] = df["future_close"] / df["close"] - 1
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    df["f1"] = rng.normal(size=len(df))
    df["f2"] = rng.normal(size=len(df))
    return df.dropna().reset_index(drop=True)

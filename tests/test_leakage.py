"""
Look-ahead and leakage guards. These are the bugs that make a backtest look
good and a live model lose money, so they are tested explicitly.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.features import FeatureEngineer
from src.models.trainer import ModelTrainer

HORIZON = 5


def trading_days_between(all_dates, a, b) -> int:
    """Number of trading days strictly between dates a and b."""
    return int(((all_dates > a) & (all_dates < b)).sum())


# --- target construction ---------------------------------------------------

def test_target_uses_future_close_of_same_symbol(ohlcv):
    df = FeatureEngineer().create_target(ohlcv.sort_values(["symbol", "date"]), horizon=HORIZON)
    for _, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        expected = g["close"].shift(-HORIZON) / g["close"] - 1
        np.testing.assert_allclose(g["target_return"], expected, equal_nan=True)
        # the last `horizon` rows of each symbol have no future -> no label
        assert g["future_close"].tail(HORIZON).isna().all()


# --- features only use the past --------------------------------------------

def _causal_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = FeatureEngineer(window=20)
    out = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for step in (fe.add_price_features, fe.add_volume_features,
                 fe.add_technical_indicators, fe.add_lag_features, fe.add_time_features):
        out = step(out)
    return out.set_index(["date", "symbol"]).sort_index()


def test_features_do_not_change_when_future_rows_are_removed(ohlcv):
    """If a feature at day t used data after t, truncating the future would change it."""
    cutoff = ohlcv["date"].sort_values().unique()[180]
    full = _causal_features(ohlcv)
    past_only = _causal_features(ohlcv[ohlcv["date"] <= cutoff])

    common = past_only.index
    pd.testing.assert_frame_equal(full.loc[common, past_only.columns], past_only, check_exact=False)


# --- purged splits ---------------------------------------------------------

def test_train_val_test_split_is_by_date_and_purged(feature_panel):
    train, val, test = ModelTrainer(mlflow_tracking=False).time_series_split(
        feature_panel, train_size=0.7, val_size=0.15, gap=HORIZON
    )
    all_dates = np.sort(feature_panel["date"].unique())

    # no calendar day is split across sets
    assert not set(train["date"]) & set(val["date"])
    assert not set(val["date"]) & set(test["date"])
    # every day keeps all its symbols
    for part in (train, val, test):
        assert (part.groupby("date").size() == feature_panel["symbol"].nunique()).all()
    # labels look HORIZON days ahead, so at least HORIZON days separate the sets
    assert trading_days_between(all_dates, train["date"].max(), val["date"].min()) >= HORIZON
    assert trading_days_between(all_dates, val["date"].max(), test["date"].min()) >= HORIZON


def test_walk_forward_folds_are_purged_and_expanding(feature_panel):
    folds, avg = ModelTrainer(mlflow_tracking=False).walk_forward_validate(
        feature_panel, model_params={"n_estimators": 20, "max_depth": 2},
        n_splits=3, gap=HORIZON,
    )
    all_dates = np.sort(feature_panel["date"].unique())
    train_starts = set()
    for f in folds:
        train_start, train_end = (pd.Timestamp(x.strip()) for x in f["train_period"].split("→"))
        test_start, _ = (pd.Timestamp(x.strip()) for x in f["test_period"].split("→"))
        assert trading_days_between(all_dates, train_end, test_start) >= HORIZON
        train_starts.add(train_start)
    assert len(train_starts) == 1, "expanding window: every fold trains from the first date"
    assert 0.0 <= avg["avg_auc"] <= 1.0


def test_noise_features_have_no_skill(feature_panel):
    """Sanity check of the evaluation itself: pure noise must score ~0.5 AUC."""
    _, avg = ModelTrainer(mlflow_tracking=False).walk_forward_validate(
        feature_panel, model_params={"n_estimators": 50, "max_depth": 2},
        n_splits=3, gap=HORIZON,
    )
    assert avg["avg_auc"] == pytest.approx(0.5, abs=0.08)

"""API contract tests. Network calls (Yahoo Finance) are replaced with synthetic data."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.data.features import FeatureEngineer
from tests.conftest import make_ohlcv

SYMBOLS = ["AAPL", "MSFT", "GOOGL"]


def _fake_external(self, df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col, val in {"vix": 18.0, "spy_return_1d": 0.0, "spy_return_5d": 0.0,
                     "xlk_return_1d": 0.0, "xlk_return_5d": 0.0, "days_to_earnings": 45}.items():
        df[col] = val
    return df


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(FeatureEngineer, "add_external_features", _fake_external)
    with TestClient(main.app) as c:
        monkeypatch.setattr(main.state.data_fetcher, "fetch",
                            lambda symbols, **_: make_ohlcv(n_days=90, symbols=symbols))
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    status = client.get("/").json()
    assert status["model_loaded"] is True
    assert 0 <= status["uptime_seconds"] < 3600


def test_backtest_summary_shape(client):
    bt = client.get("/backtest").json()
    assert {"base_rate", "strategies", "equity", "permutation_test", "ic_test"} <= bt.keys()
    assert 0 < bt["base_rate"] < 1


def test_dashboard_serves_versioned_assets(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"
    assert "app.js?v=" in r.text


def test_predict_contract(client):
    r = client.post("/predict", json={"symbols": SYMBOLS})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert sorted(p["symbol"] for p in preds) == sorted(SYMBOLS)
    for p in preds:
        assert 0 <= p["p_up"] <= 1
        assert p["edge"] == pytest.approx(p["p_up"] - p["base_rate"])
        assert 1 <= p["rank"] <= len(SYMBOLS)
        assert p["signal"] in {"no edge", "weak tilt", "moderate tilt"}


def test_signal_label_thresholds():
    assert main.signal_label(0.01) == "no edge"
    assert main.signal_label(-0.05) == "weak tilt"
    assert main.signal_label(0.10) == "moderate tilt"

"""Backtest engine: returns, costs, risk metrics and significance tests."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import (
    buy_and_hold, long_flat, long_short_rank, max_drawdown, performance_stats,
    permutation_test_auc, permutation_test_ic, run_strategy, sharpe,
)

H = 5


def make_preds(n_dates=200, symbols=("A", "B", "C", "D", "E"), seed=0, skill=0.0):
    """Synthetic OOS predictions. `skill` mixes the true forward return into the score."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for d in dates:
        market = rng.normal(0, 0.01)
        for s in symbols:
            ret = market + rng.normal(0, 0.02)
            score = 0.5 + skill * np.sign(ret) * 0.1 + rng.normal(0, 0.05)
            rows.append({"date": d, "symbol": s, "target_return": ret,
                         "target_direction": int(ret > 0), "score": score})
    return pd.DataFrame(rows)


# --- metrics ---------------------------------------------------------------

def test_max_drawdown_known_path():
    # equity 1.0 -> 1.1 -> 0.88 -> 0.968; worst peak-to-trough is 1.1 -> 0.88 = -20%
    assert max_drawdown(pd.Series([0.10, -0.20, 0.10])) == pytest.approx(-0.20)


def test_sharpe_zero_volatility_is_zero_not_inf():
    assert sharpe(np.full(10, 0.01), periods_per_year=52) == 0.0


def test_sharpe_annualisation():
    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(52)
    assert sharpe(r, 52) == pytest.approx(expected)


# --- strategies & costs ----------------------------------------------------

def test_rebalances_do_not_overlap():
    preds = make_preds(n_dates=50)
    res = run_strategy(preds, buy_and_hold, horizon=H, cost_bps=0)
    gaps = np.diff(np.sort(preds["date"].unique()).searchsorted(res.index.values))
    assert (gaps == H).all()


def test_costs_charged_on_turnover_only():
    preds = make_preds(n_dates=50)
    gross = run_strategy(preds, buy_and_hold, horizon=H, cost_bps=0)
    net = run_strategy(preds, buy_and_hold, horizon=H, cost_bps=10)
    # buy & hold: full turnover (1.0) on entry, zero afterwards
    assert net["turnover"].iloc[0] == pytest.approx(1.0)
    assert (net["turnover"].iloc[1:] == 0).all()
    assert gross["net"].iloc[0] - net["net"].iloc[0] == pytest.approx(10 / 1e4)
    np.testing.assert_allclose(gross["net"].iloc[1:], net["net"].iloc[1:])


def test_long_short_is_market_neutral():
    day = make_preds(n_dates=1)
    w = long_short_rank("score", k=2)(day)
    assert w.sum() == pytest.approx(0.0)
    assert w.abs().sum() == pytest.approx(1.0)


def test_long_flat_holds_cash_below_threshold():
    day = make_preds(n_dates=1)
    assert long_flat("score", threshold=10.0)(day).sum() == 0.0


def test_performance_stats_keys():
    res = run_strategy(make_preds(), buy_and_hold, horizon=H, cost_bps=10)
    stats = performance_stats(res, horizon=H)
    assert {"cagr", "sharpe", "max_drawdown", "hit_rate", "avg_exposure"} <= stats.keys()
    assert stats["max_drawdown"] <= 0


# --- significance tests ----------------------------------------------------

def test_auc_permutation_detects_real_signal():
    res = permutation_test_auc(make_preds(skill=1.0), "score", horizon=H, n_perm=200)
    assert res["auc"] > 0.7 and res["p_value"] < 0.01


def test_auc_permutation_false_positive_rate_on_noise():
    """On pure noise a 5% test should fire ~5% of the time - not much more."""
    p = [permutation_test_auc(make_preds(n_dates=100, skill=0.0, seed=s), "score",
                              horizon=H, n_perm=100)["p_value"] for s in range(40)]
    assert np.mean(np.array(p) < 0.05) <= 0.15


def test_ic_test_ignores_market_timing():
    """A score that only knows the market's direction (same for every stock) has zero
    stock-picking skill, so the within-date IC test must not reward it."""
    preds = make_preds(seed=4)
    preds["score"] = preds.groupby("date")["target_return"].transform("mean")
    res = permutation_test_ic(preds, "score", horizon=H, n_perm=100)
    assert res["mean_ic"] == 0.0 and res["p_value"] > 0.05


def test_ic_test_detects_stock_picking():
    res = permutation_test_ic(make_preds(skill=1.0, seed=5), "score", horizon=H, n_perm=200)
    assert res["mean_ic"] > 0.2 and res["p_value"] < 0.01

"""
Out-of-sample backtest: turns model probabilities into portfolios and compares
them against naive baselines, after transaction costs.

Pipeline
--------
1. ``generate_oos_predictions`` — purged, date-based walk-forward. Every row in the
   result was predicted by a model that never saw its date (or any label that
   overlaps it).
2. ``build_weights`` — converts scores into per-stock weights on non-overlapping
   rebalance dates (one every ``horizon`` trading days).
3. ``run_strategy`` / ``performance_stats`` — net-of-cost returns, Sharpe, drawdown.
4. ``permutation_test_auc`` / ``bootstrap_sharpe_diff`` — is the edge real?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.xgboost_model import XGBoostModel
from src.utils.logger import log

NON_FEATURE_COLS = [
    'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'future_close', 'target_return', 'target_direction',
]


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# ---------------------------------------------------------------------------
# 1. Purged walk-forward predictions
# ---------------------------------------------------------------------------

def generate_oos_predictions(
    df: pd.DataFrame,
    xgb_params: Optional[Dict] = None,
    n_splits: int = 5,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward split on *unique dates* (not rows), so all
    symbols for a given day land in the same fold. A gap of ``horizon`` trading
    days separates train from test: the last training labels look ``horizon``
    days ahead, so without the gap they would overlap the test period.

    Returns one row per (date, symbol) in the test folds with columns
    ``p_xgb``, ``p_logreg`` and ``s_momentum`` (raw 5-day return, a score).
    """
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    feats = feature_columns(df)
    dates = np.array(sorted(df['date'].unique()))
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=horizon)

    out = []
    for fold, (tr_d, te_d) in enumerate(tscv.split(dates), 1):
        train_dates, test_dates = dates[tr_d], dates[te_d]

        # Early-stopping split inside the training window, also purged.
        cut = int(len(train_dates) * 0.85)
        fit_dates = train_dates[: cut - horizon]
        val_dates = train_dates[cut:]

        fit = df[df['date'].isin(fit_dates)]
        val = df[df['date'].isin(val_dates)]
        trn = df[df['date'].isin(train_dates)]
        tst = df[df['date'].isin(test_dates)].copy()

        xgb = XGBoostModel(params=xgb_params)
        xgb.train(fit[feats], fit['target_direction'],
                  val[feats], val['target_direction'],
                  early_stopping_rounds=30, verbose=False)
        tst['p_xgb'] = xgb.predict_proba(tst[feats])[:, 1]

        logreg = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        logreg.fit(trn[feats], trn['target_direction'])
        tst['p_logreg'] = logreg.predict_proba(tst[feats])[:, 1]

        tst['s_momentum'] = tst['return_5d']
        tst['fold'] = fold
        out.append(tst)

        log.info(f"Fold {fold}: train {pd.Timestamp(train_dates[0]).date()} → "
                 f"{pd.Timestamp(train_dates[-1]).date()}, test "
                 f"{pd.Timestamp(test_dates[0]).date()} → {pd.Timestamp(test_dates[-1]).date()}")

    keep = ['date', 'symbol', 'fold', 'target_return', 'target_direction',
            'vix', 'p_xgb', 'p_logreg', 's_momentum']
    return pd.concat(out)[[c for c in keep if c in out[0].columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Scores -> weights
# ---------------------------------------------------------------------------

def rebalance_dates(preds: pd.DataFrame, horizon: int) -> np.ndarray:
    """Every ``horizon``-th OOS date, so consecutive holding periods don't overlap."""
    dates = np.array(sorted(preds['date'].unique()))
    return dates[::horizon]


def long_flat(score_col: str, threshold: float) -> Callable[[pd.DataFrame], pd.Series]:
    """Equal-weight long every stock whose score beats the threshold; cash otherwise."""
    def f(day: pd.DataFrame) -> pd.Series:
        n = len(day)
        return pd.Series(np.where(day[score_col] > threshold, 1.0 / n, 0.0), index=day['symbol'])
    return f


def buy_and_hold(day: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / len(day), index=day['symbol'])


def long_short_rank(score_col: str, k: int = 2) -> Callable[[pd.DataFrame], pd.Series]:
    """Market-neutral: long the top-k scores, short the bottom-k, 50/50 gross."""
    def f(day: pd.DataFrame) -> pd.Series:
        ranked = day.sort_values(score_col)
        w = pd.Series(0.0, index=ranked['symbol'])
        w.iloc[-k:] = 0.5 / k
        w.iloc[:k] = -0.5 / k
        return w
    return f


# ---------------------------------------------------------------------------
# 3. Returns and stats
# ---------------------------------------------------------------------------

def run_strategy(
    preds: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame], pd.Series],
    horizon: int,
    cost_bps: float,
) -> pd.DataFrame:
    """
    Per-period gross/net returns. Decision uses features at close t; the position
    earns the close(t) → close(t+horizon) return. Cost = |Δweight| × cost_bps,
    charged on every rebalance (weights drift is ignored — a small approximation).
    """
    rows, prev_w = [], pd.Series(dtype=float)
    for d in rebalance_dates(preds, horizon):
        day = preds[preds['date'] == d]
        w = weight_fn(day)
        r = day.set_index('symbol')['target_return'].reindex(w.index).fillna(0.0)
        gross = float((w * r).sum())
        turnover = float(w.sub(prev_w, fill_value=0.0).abs().sum())
        cost = turnover * cost_bps / 1e4
        rows.append({'date': d, 'gross': gross, 'net': gross - cost,
                     'turnover': turnover, 'exposure': float(w.abs().sum())})
        prev_w = w
    return pd.DataFrame(rows).set_index('date')


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def performance_stats(res: pd.DataFrame, horizon: int, col: str = 'net') -> Dict[str, float]:
    r = res[col]
    ppy = 252 / horizon
    years = len(r) / ppy
    total = float((1 + r).prod())
    return {
        'cagr': total ** (1 / years) - 1 if years > 0 else 0.0,
        'ann_vol': float(r.std(ddof=1) * np.sqrt(ppy)),
        'sharpe': sharpe(r.values, ppy),
        'max_drawdown': max_drawdown(r),
        'hit_rate': float((r > 0).mean()),
        'avg_turnover': float(res['turnover'].mean()),
        'avg_exposure': float(res['exposure'].mean()),
        'n_periods': int(len(r)),
    }


def classification_stats(preds: pd.DataFrame, score_col: str, threshold: float) -> Dict[str, float]:
    y = preds['target_direction']
    s = preds[score_col]
    return {
        'accuracy': float(accuracy_score(y, (s > threshold).astype(int))),
        'auc': float(roc_auc_score(y, s)),
    }


# ---------------------------------------------------------------------------
# 4. Is it real?
# ---------------------------------------------------------------------------

def permutation_test_auc(
    preds: pd.DataFrame, score_col: str, horizon: int, n_perm: int = 1000, seed: int = 0,
) -> Dict[str, float]:
    """
    Overall skill. H0: the score has no information about direction (AUC = 0.5).
    Uses non-overlapping rebalance dates only (overlapping 5-day labels are
    autocorrelated and would understate p); labels are shuffled across all of them.
    """
    rng = np.random.default_rng(seed)
    sub = preds[preds['date'].isin(rebalance_dates(preds, horizon))]
    y = sub['target_direction'].values
    s = sub[score_col].values
    observed = roc_auc_score(y, s)
    null = np.array([roc_auc_score(rng.permutation(y), s) for _ in range(n_perm)])
    return {'auc': float(observed), 'null_mean': float(null.mean()),
            'p_value': float((np.sum(null >= observed) + 1) / (n_perm + 1)),
            'n_obs': int(len(y))}


def _mean_rank_ic(score: np.ndarray, ret: np.ndarray, groups) -> float:
    ics = []
    for idx in groups:
        s, r = pd.Series(score[idx]).rank(), pd.Series(ret[idx]).rank()
        if s.nunique() > 1 and r.nunique() > 1:
            ics.append(np.corrcoef(s, r)[0, 1])
    return float(np.mean(ics)) if ics else 0.0


def permutation_test_ic(
    preds: pd.DataFrame, score_col: str, horizon: int, n_perm: int = 1000, seed: int = 0,
) -> Dict[str, float]:
    """
    Stock-picking skill. Statistic: mean cross-sectional rank IC (Spearman
    correlation between score and forward return across stocks, per rebalance
    date). H0: IC = 0. Returns are shuffled *within each date*, so market-wide
    up/down weeks can't masquerade as skill. Dates where the model gives every
    stock the same score carry no ranking and are skipped.
    """
    rng = np.random.default_rng(seed)
    sub = preds[preds['date'].isin(rebalance_dates(preds, horizon))].reset_index(drop=True)
    s, r = sub[score_col].values, sub['target_return'].values
    groups = list(sub.groupby('date').indices.values())
    observed = _mean_rank_ic(s, r, groups)

    null = np.empty(n_perm)
    for i in range(n_perm):
        r_perm = r.copy()
        for idx in groups:
            r_perm[idx] = rng.permutation(r_perm[idx])
        null[i] = _mean_rank_ic(s, r_perm, groups)

    return {'mean_ic': observed, 'null_mean': float(null.mean()),
            'p_value': float((np.sum(null >= observed) + 1) / (n_perm + 1)),
            'n_dates': len(groups)}


def bootstrap_sharpe_diff(
    a: pd.Series, b: pd.Series, horizon: int, block: int = 4,
    n_boot: int = 2000, seed: int = 0,
) -> Dict[str, float]:
    """
    Moving-block bootstrap of Sharpe(a) − Sharpe(b), resampling the *same*
    blocks for both series so their correlation is kept.
    """
    rng = np.random.default_rng(seed)
    a, b = a.values, b.values
    n, ppy = len(a), 252 / horizon
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        diffs[i] = sharpe(a[idx], ppy) - sharpe(b[idx], ppy)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'diff': sharpe(a, ppy) - sharpe(b, ppy), 'ci_low': float(lo),
            'ci_high': float(hi), 'p_le_zero': float((diffs <= 0).mean())}


def regime_breakdown(preds: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    """Accuracy / AUC split by VIX tercile (low / mid / high fear)."""
    if 'vix' not in preds.columns:
        return pd.DataFrame()
    p = preds.copy()
    p['vix_regime'] = pd.qcut(p['vix'], 3, labels=['low VIX', 'mid VIX', 'high VIX'])
    rows = []
    for regime, g in p.groupby('vix_regime', observed=True):
        rows.append({'regime': str(regime), 'n': len(g),
                     'base_rate_up': float(g['target_direction'].mean()),
                     **classification_stats(g, score_col, threshold)})
    return pd.DataFrame(rows)


@dataclass
class StrategySpec:
    name: str
    weight_fn: Callable[[pd.DataFrame], pd.Series]

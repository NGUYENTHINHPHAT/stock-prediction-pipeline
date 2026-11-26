"""
Backtest the model against naive baselines, net of transaction costs.

Answers the questions an interviewer actually asks:
  * Does the model beat "always predict UP" and simple momentum?
  * Does the edge survive transaction costs?
  * Is it statistically distinguishable from luck?
  * In which market regime does it work?

Outputs (in results/):
  backtest_report.md   — tables ready to paste into the README
  backtest_summary.json
  equity_curve.png

Usage:
    python backtest.py
    python backtest.py --cost-bps 10 --folds 5 --threshold 0.55
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.backtest import (
    StrategySpec, buy_and_hold, bootstrap_sharpe_diff, classification_stats,
    generate_oos_predictions, long_flat, long_short_rank, performance_stats,
    permutation_test_auc, permutation_test_ic, regime_breakdown, run_strategy,
)
from src.utils.config import settings
from src.utils.logger import log
from validate import load_data

RESULTS = Path("results")


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main(symbols, n_splits, cost_bps, threshold, params_file):
    H = settings.prediction_horizon
    RESULTS.mkdir(exist_ok=True)

    params = None
    if Path(params_file).exists():
        params = json.loads(Path(params_file).read_text())
        log.info(f"Using XGBoost params from {params_file}")

    df = load_data(symbols)
    preds = generate_oos_predictions(df, xgb_params=params, n_splits=n_splits, horizon=H)
    preds.to_csv(RESULTS / "oos_predictions.csv", index=False)

    oos_start, oos_end = preds['date'].min().date(), preds['date'].max().date()
    log.info(f"Out-of-sample: {oos_start} → {oos_end}, {len(preds)} predictions")

    # ---- classification vs baselines --------------------------------------
    base_rate = float(preds['target_direction'].mean())
    clf_rows = [
        {'model': 'Always UP (baseline)', 'accuracy': base_rate, 'auc': 0.5},
        {'model': 'Momentum: sign of 5d return', **classification_stats(preds, 's_momentum', 0.0)},
        {'model': 'Logistic regression', **classification_stats(preds, 'p_logreg', 0.5)},
        {'model': 'XGBoost (tuned)', **classification_stats(preds, 'p_xgb', 0.5)},
    ]
    clf = pd.DataFrame(clf_rows)

    # ---- strategies --------------------------------------------------------
    specs = [
        StrategySpec('Buy & hold (equal weight)', buy_and_hold),
        StrategySpec('Momentum long/flat', long_flat('s_momentum', 0.0)),
        StrategySpec('LogReg long/flat', long_flat('p_logreg', threshold)),
        StrategySpec('XGBoost long/flat', long_flat('p_xgb', threshold)),
        StrategySpec('XGBoost long/short rank (mkt-neutral)', long_short_rank('p_xgb', k=2)),
    ]
    runs = {s.name: run_strategy(preds, s.weight_fn, H, cost_bps) for s in specs}
    strat = pd.DataFrame([{'strategy': n, **performance_stats(r, H)} for n, r in runs.items()])
    gross_sharpe = {n: performance_stats(r, H, col='gross')['sharpe'] for n, r in runs.items()}
    strat.insert(1, 'sharpe_gross', strat['strategy'].map(gross_sharpe))

    # ---- cost sensitivity for the main model ------------------------------
    cost_rows = []
    for bps in [0, 5, 10, 20]:
        r = run_strategy(preds, long_flat('p_xgb', threshold), H, bps)
        cost_rows.append({'cost_bps': bps, **{k: performance_stats(r, H)[k] for k in ['cagr', 'sharpe']}})
    costs = pd.DataFrame(cost_rows)

    # ---- significance ------------------------------------------------------
    perm = permutation_test_auc(preds, 'p_xgb', H)
    ic = permutation_test_ic(preds, 'p_xgb', H)
    boot = bootstrap_sharpe_diff(runs['XGBoost long/flat']['net'],
                                 runs['Buy & hold (equal weight)']['net'], H)
    regimes = regime_breakdown(preds, 'p_xgb', 0.5)

    # ---- equity curve ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in runs.items():
        ax.plot(pd.to_datetime(r.index), (1 + r['net']).cumprod(), label=name,
                lw=2.2 if name.startswith('XGBoost long/flat') else 1.3)
    ax.axhline(1, color='grey', lw=0.8, ls=':')
    ax.set_title(f"Out-of-sample equity, net of {cost_bps:g} bps costs "
                 f"({H}-day rebalance, {', '.join(symbols)})")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "equity_curve.png", dpi=130)

    # ---- report ------------------------------------------------------------
    pct_cols = ['cagr', 'ann_vol', 'max_drawdown', 'hit_rate']
    strat_md = strat.copy()
    for c in pct_cols:
        strat_md[c] = strat_md[c].map(fmt_pct)
    for c in ['sharpe', 'sharpe_gross', 'avg_turnover', 'avg_exposure']:
        strat_md[c] = strat_md[c].map(lambda x: f"{x:.2f}")
    clf_md = clf.assign(accuracy=clf['accuracy'].map(fmt_pct), auc=clf['auc'].map(lambda x: f"{x:.3f}"))
    costs_md = costs.assign(cagr=costs['cagr'].map(fmt_pct), sharpe=costs['sharpe'].map(lambda x: f"{x:.2f}"))

    lines = [
        "# Backtest results",
        "",
        f"Out-of-sample period: **{oos_start} → {oos_end}** · {len(symbols)} stocks · "
        f"{n_splits}-fold purged walk-forward · {H}-day horizon, non-overlapping rebalances · "
        f"long/flat threshold P(up) > {threshold} · costs {cost_bps:g} bps per unit turnover.",
        "",
        "## Classification vs baselines (all OOS rows)",
        "",
        clf_md.to_markdown(index=False),
        "",
        "## Strategy performance (net of costs unless noted)",
        "",
        strat_md.to_markdown(index=False),
        "",
        "![Equity curve](equity_curve.png)",
        "",
        "## Cost sensitivity — XGBoost long/flat",
        "",
        costs_md.to_markdown(index=False),
        "",
        "## Is the edge real?",
        "",
        f"- **Overall skill: permutation test on AUC ({perm['n_obs']} non-overlapping obs):** "
        f"AUC = {perm['auc']:.3f} vs null {perm['null_mean']:.3f}, **p = {perm['p_value']:.3f}**",
        f"- **Stock-picking skill: mean weekly rank IC ({ic['n_dates']} dates, returns shuffled within each date):** "
        f"IC = {ic['mean_ic']:+.3f} vs null {ic['null_mean']:+.3f}, **p = {ic['p_value']:.3f}**",
        f"- **Block bootstrap, Sharpe(XGBoost long/flat) − Sharpe(buy & hold):** "
        f"{boot['diff']:+.2f}, 95% CI [{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]",
        "",
    ]
    if not regimes.empty:
        reg_md = regimes.assign(base_rate_up=regimes['base_rate_up'].map(fmt_pct),
                                accuracy=regimes['accuracy'].map(fmt_pct),
                                auc=regimes['auc'].map(lambda x: f"{x:.3f}"))
        lines += ["## By volatility regime (XGBoost)", "", reg_md.to_markdown(index=False), ""]

    # Equity curves for the dashboard (growth of $1, net of costs).
    dash_names = ['Buy & hold (equal weight)', 'XGBoost long/flat',
                  'XGBoost long/short rank (mkt-neutral)']
    equity = {
        'dates': [str(pd.Timestamp(d).date()) for d in runs[dash_names[0]].index],
        'series': {n: [round(float(v), 4) for v in (1 + runs[n]['net']).cumprod()] for n in dash_names},
    }
    base_rate_by_symbol = preds.groupby('symbol')['target_direction'].mean().round(4).to_dict()

    (RESULTS / "backtest_report.md").write_text("\n".join(lines))
    (RESULTS / "backtest_summary.json").write_text(json.dumps({
        'oos_period': [str(oos_start), str(oos_end)], 'cost_bps': cost_bps, 'threshold': threshold,
        'horizon_days': H, 'symbols': list(symbols), 'n_folds': n_splits,
        'base_rate': base_rate, 'base_rate_by_symbol': base_rate_by_symbol, 'equity': equity,
        'classification': clf_rows, 'strategies': strat.to_dict('records'),
        'cost_sensitivity': costs.to_dict('records'), 'permutation_test': perm, 'ic_test': ic,
        'sharpe_diff_bootstrap': boot, 'regimes': regimes.to_dict('records'),
    }, indent=2, default=str))

    print("\n".join(lines))
    log.info(f"Saved report to {RESULTS}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--params-file", default="models/best_params.json")
    a = p.parse_args()
    main(a.symbols or settings.symbol_list, a.folds, a.cost_bps, a.threshold, a.params_file)

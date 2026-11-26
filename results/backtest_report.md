# Backtest results

Out-of-sample period: **2022-11-08 → 2026-11-17** · 5 stocks · 5-fold purged walk-forward · 5-day horizon, non-overlapping rebalances · long/flat threshold P(up) > 0.55 · costs 10 bps per unit turnover.

## Classification vs baselines (all OOS rows)

| model                       | accuracy   |   auc |
|:----------------------------|:-----------|------:|
| Always UP (baseline)        | 55.2%      | 0.5   |
| Momentum: sign of 5d return | 50.6%      | 0.495 |
| Logistic regression         | 48.5%      | 0.492 |
| XGBoost (tuned)             | 49.4%      | 0.496 |

## Strategy performance (net of costs unless noted)

| strategy                              |   sharpe_gross | cagr   | ann_vol   |   sharpe | max_drawdown   | hit_rate   |   avg_turnover |   avg_exposure |   n_periods |
|:--------------------------------------|---------------:|:-------|:----------|---------:|:---------------|:-----------|---------------:|---------------:|------------:|
| Buy & hold (equal weight)             |           1.22 | 30.9%  | 24.5%     |     1.22 | -26.2%         | 55.0%      |           0    |           1    |         202 |
| Momentum long/flat                    |           0.88 | 10.6%  | 15.6%     |     0.72 | -18.9%         | 43.6%      |           0.5  |           0.56 |         202 |
| LogReg long/flat                      |           0.47 | 2.5%   | 9.6%      |     0.31 | -15.3%         | 32.7%      |           0.3  |           0.34 |         202 |
| XGBoost long/flat                     |           0.52 | 1.6%   | 4.4%      |     0.39 | -8.4%          | 18.3%      |           0.11 |           0.1  |         202 |
| XGBoost long/short rank (mkt-neutral) |          -0.11 | -5.7%  | 13.1%     |    -0.39 | -33.2%         | 49.5%      |           0.71 |           1    |         202 |

![Equity curve](equity_curve.png)

## Cost sensitivity — XGBoost long/flat

|   cost_bps | cagr   |   sharpe |
|-----------:|:-------|---------:|
|          0 | 2.2%   |     0.52 |
|          5 | 1.9%   |     0.45 |
|         10 | 1.6%   |     0.39 |
|         20 | 1.1%   |     0.26 |

## Is the edge real?

- **Overall skill: permutation test on AUC (1010 non-overlapping obs):** AUC = 0.498 vs null 0.500, **p = 0.543**
- **Stock-picking skill: mean weekly rank IC (202 dates, returns shuffled within each date):** IC = -0.008 vs null +0.003, **p = 0.620**
- **Block bootstrap, Sharpe(XGBoost long/flat) − Sharpe(buy & hold):** -0.83, 95% CI [-2.02, +0.28]

## By volatility regime (XGBoost)

| regime   |    n | base_rate_up   | accuracy   |   auc |
|:---------|-----:|:---------------|:-----------|------:|
| low VIX  | 1690 | 53.8%          | 51.1%      | 0.515 |
| mid VIX  | 1685 | 53.4%          | 49.1%      | 0.474 |
| high VIX | 1675 | 58.6%          | 48.1%      | 0.503 |

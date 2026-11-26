---
title: Tech Stock Prediction with Technical Analysis
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Tech Stock Prediction with Technical Analysis

> **Live Demo:** [Stock Prediction Dashboard](https://PhatNguyen39-Stock-Prediction-Pipeline.hf.space/dashboard)

> **Disclaimer:** This project is for **demonstration and portfolio purposes only**. **This is not financial advice and should not be used for trading decisions.**

Production-ready ML pipeline for tech stock price direction prediction using XGBoost, with a full experiment tracking and hyperparameter tuning workflow.

## Scope

The model is trained on **5 large-cap tech stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA using **5 years** of historical data.

Predictions are **5 trading days ahead** (weekly direction).

Features include both technical indicators and external market signals (VIX, SPY, XLK sector ETF, earnings proximity). The project's focus is **honest evaluation**: a purged walk-forward backtest with transaction costs, naive baselines, and significance tests. The conclusion is that these features do **not** carry a statistically significant edge on large-cap tech. See [Backtest Results](#-backtest-results).

## 🎯 Features

- **Automated Data Pipeline**: Fetches OHLCV data from Yahoo Finance (5 years lookback)
- **Feature Engineering**: 47+ features across technical indicators, external market signals, and time features
- **External Market Signals**: VIX fear index, SPY/XLK returns, days to next earnings
- **Hyperparameter Tuning**: Optuna-based search with 50+ trials, saves best params automatically
- **Walk-Forward Validation**: 5-fold expanding-window cross-validation for honest performance estimates
- **Backtest with Costs**: Long/flat and market-neutral strategies vs buy-and-hold, momentum and logistic-regression baselines — Sharpe, drawdown, turnover, cost sensitivity
- **Significance Testing**: Permutation tests for overall skill (AUC) and stock-picking skill (weekly rank IC), plus a block-bootstrap CI on the Sharpe difference
- **Research Dashboard**: Shows the backtest verdict, equity curves and baselines next to live predictions, with each P(up) set against the stock's historical base rate
- **Time-Series Safe**: Chronological train/val/test split, no data leakage
- **XGBoost Classifier**: Binary classification (UP/DOWN), probability output with confidence labels
- **Web Dashboard**: Built-in browser UI — pick from the 5 supported stocks
- **FastAPI Serving**: Production REST API, prediction uses only 90 days of data (fast)
- **MLflow Tracking**: Full experiment history — params, metrics, feature importance per run
- **Docker Support**: Containerized deployment
- **Hugging Face Spaces**: Free cloud deployment with Docker SDK

## 📁 Project Structure

```
stock-prediction-pipeline/
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Data fetching (Yahoo Finance)
│   │   ├── features.py         # Feature engineering + external signals
│   │   └── validator.py        # Data validation & cleaning
│   ├── models/
│   │   ├── base.py             # Abstract model interface
│   │   ├── xgboost_model.py    # XGBoost implementation
│   │   ├── ensemble.py         # Ensemble framework (ready for extension)
│   │   └── trainer.py          # Training + walk-forward validation
│   ├── api/
│   │   ├── main.py             # FastAPI application
│   │   ├── templates/
│   │   │   └── dashboard.html  # Web dashboard UI
│   │   └── static/
│   │       ├── css/style.css
│   │       └── js/app.js
│   └── utils/
│       ├── config.py           # Configuration (lookback, horizon, symbols)
│       └── logger.py           # Logging setup
├── models/
│   ├── saved/
│   │   └── latest_model.pkl    # Active model loaded by the API
│   └── best_params.json        # Best hyperparams from last tune.py run
├── data/                       # Data storage (gitignored)
├── logs/                       # Application logs (gitignored)
├── mlruns/                     # MLflow experiment history (gitignored)
├── train.py                    # Train with best_params.json (auto-loaded)
├── tune.py                     # Optuna hyperparameter search + retrain
├── validate.py                 # Walk-forward cross-validation
├── check_importance.py         # Print feature importance from saved model
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repo-url>
cd stock-prediction-pipeline

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Recommended Workflow

#### Option A — Tune then train (best results)

```bash
# Step 1: Search for best hyperparameters (50 trials, ~10–20 min)
python tune.py

# Step 2: Validate performance across 5 market regimes
python validate.py

# Step 3: Train is not required — tune.py already saves the model.
#         Re-run train.py only if you want to retrain on fresh data
#         using the saved best params:
python train.py
```

#### Option B — Train directly with defaults

```bash
python train.py
```

`train.py` automatically loads `models/best_params.json` if it exists.
If the file is missing or empty, it falls back to XGBoost defaults.

### 3. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Open the Dashboard

```
http://localhost:8000/dashboard
```

From the dashboard you can:
- **See model status** — indicator shows whether a model is loaded
- **Train the model** — click "Train Model" and poll until complete
- **Get predictions** — toggle stocks, hit submit, see direction + probability + confidence badge

Only `medium` and `high` confidence predictions (probability > 60%) are worth acting on.

### 5. Make Predictions via API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"]}'
```

Response:
```json
{
  "predictions": [
    {
      "symbol": "AAPL",
      "current_price": 185.23,
      "predicted_direction": 1,
      "probability": 0.6234,
      "confidence": "medium",
      "timestamp": "2026-05-06T10:30:00"
    }
  ],
  "model_name": "xgboost",
  "model_last_trained": "2026-05-06T09:00:00"
}
```

**Confidence labels:**
- `high` → probability > 70%
- `medium` → probability 60–70%
- `low` → probability ≤ 60% — ignore these signals

## 🔧 Scripts

### `tune.py` — Hyperparameter search

```bash
python tune.py                            # 50 trials, optimise val_precision
python tune.py --trials 100               # more trials = better search
python tune.py --metric val_auc           # optimise AUC instead
python tune.py --symbols AAPL MSFT        # specific symbols
```

Runs Optuna search, saves best params to `models/best_params.json`, retrains the final model, saves to `models/saved/latest_model.pkl`.

### `train.py` — Training

```bash
python train.py                           # auto-loads best_params.json
python train.py --symbols AAPL MSFT TSLA  # specific symbols
python train.py --params-file other.json  # explicit params file
```

### `validate.py` — Walk-forward validation

```bash
python validate.py           # 5 folds with best_params.json
python validate.py --folds 8 # more folds
python validate.py --no-params  # XGBoost defaults
```

Reports per-fold and average accuracy/precision/AUC with std across market regimes. Does **not** save a model — purely diagnostic.

### `backtest.py` — Strategy backtest vs baselines

```bash
python backtest.py                                  # 10 bps costs, P(up) > 0.55
python backtest.py --cost-bps 5 --threshold 0.6
```

Purged walk-forward predictions → weekly non-overlapping rebalances → net-of-cost returns. Writes `results/backtest_report.md`, `results/backtest_summary.json` and `results/equity_curve.png`.

### `check_importance.py` — Feature importance

```bash
python check_importance.py
```

Prints all features ranked by XGBoost importance from the current `latest_model.pkl`.

## 🧠 Understanding the Pipeline

### Data Flow

```
Yahoo Finance (5 years OHLCV)
        ↓
Validation & Cleaning
        ↓
Feature Engineering (47+ features)
        ↓
Walk-forward / Train-Val-Test Split
        ↓
XGBoost Binary Classifier
        ↓
REST API → Dashboard
```

### Features (47+)

**Price** — 1d/5d/10d/20d returns, momentum, HL spread, close position in range

**Volume** — volume change, volume/MA ratio, Volume-Price Trend (VPT)

**Technical Indicators** — SMA(20), EMA(20), MACD, RSI(14), Bollinger Bands (width + position), Stochastic K/D, ATR(14)

**Lag Features** — close, volume, return lagged at [1, 2, 3, 5, 10] days

**Time** — day of week (+ cyclical encoding), day of month, month, quarter

**External Market Signals** (fetched live during training and prediction):
- `vix` — CBOE VIX fear index (market-wide risk regime)
- `spy_return_1d`, `spy_return_5d` — broad market momentum
- `xlk_return_1d`, `xlk_return_5d` — tech sector momentum
- `days_to_earnings` — days until next earnings announcement (0–90, capped)

### Prediction at Inference Time

Training uses 5 years of data to learn patterns. Predicting uses only the **last 90 days** to compute feature values for the most recent row — fast and efficient.

### Preventing Data Leakage

- Chronological split only — no shuffling
- **Purged splits by date, not by row** (`backtest.py`): all symbols for a day land in the same fold, and a gap of `horizon` trading days separates train from test, so no training label overlaps the test period
- In the stock-picking (rank IC) test, returns are shuffled only *within* each date, so market-wide up/down weeks can't masquerade as skill
- External features fetched using only past data at each point in time

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/dashboard` | Web UI |
| POST | `/predict` | Get predictions for symbols |
| POST | `/train` | Trigger model training |
| GET | `/training/status` | Poll training progress |
| GET | `/model/info` | Current model metadata |

## 📈 MLflow Tracking

Every `train.py` and `tune.py` run is automatically logged.

```bash
mlflow ui --port 5001
# Open http://localhost:5001
```

> **macOS note:** Port 5000 is used by AirPlay Receiver. Use `--port 5001`.

Each run stores:
- **Params**: model type, feature count, data sizes
- **Metrics**: accuracy, precision, AUC for train / val / test splits
- **Artifacts**: `feature_importance.csv`, trained model pickle

## 📉 Backtest Results

Out-of-sample **Nov 2022 → Nov 2026**, 5 stocks, 5-fold purged walk-forward, 5-day horizon, weekly non-overlapping rebalances, **10 bps** cost per unit turnover. Full report: [`results/backtest_report.md`](results/backtest_report.md).

![Out-of-sample equity curves](results/equity_curve.png)

**Does the model beat naive baselines?**

| Model | Accuracy | AUC |
|---|---|---|
| Always predict UP | **55.2%** | 0.500 |
| Momentum (sign of 5d return) | 50.6% | 0.495 |
| Logistic regression | 48.5% | 0.492 |
| XGBoost (tuned) | 49.4% | 0.496 |

**Does it make money after costs?**

| Strategy | Net Sharpe | CAGR | Max DD | Avg exposure |
|---|---|---|---|---|
| Buy & hold (equal weight) | **1.22** | **30.9%** | -26.2% | 100% |
| Momentum long/flat | 0.72 | 10.6% | -18.9% | 56% |
| LogReg long/flat | 0.31 | 2.5% | -15.3% | 34% |
| XGBoost long/flat (P > 0.55) | 0.39 | 1.6% | **-8.4%** | 10% |
| XGBoost long/short rank | -0.39 | -5.7% | -33.2% | 100% |

**Is the edge real?**
- **Overall skill.** Permutation test on AUC over non-overlapping weeks: AUC 0.498, **p = 0.54**.
- **Stock-picking skill.** Mean weekly rank IC (Spearman correlation of score vs. forward return across the 5 stocks), with returns shuffled within each date: IC −0.008, **p = 0.62**.
- **Block bootstrap**, Sharpe(XGBoost long/flat) − Sharpe(buy & hold): **−0.83, 95% CI [−2.02, +0.28]**.
- No VIX regime (low / mid / high) shows AUC meaningfully above 0.5.

### Interpretation

- **The model has no detectable edge.** It loses to "always predict UP" on accuracy, and its AUC is within noise of 0.5. In a 4-year bull market for mega-cap tech, the base rate *is* the signal.
- **The model mostly sits in cash.** It's invested only 10% of the time, which explains its small drawdown. That isn't skill: it earns 1.6% a year against 30.9% for buy & hold.
- **The market-neutral version is the cleanest test of stock-picking skill**, since it removes market direction. It loses money, which matches the slightly negative rank IC.
- **Tuning can't rescue it.** Retuning on 5 years of data, early stopping keeps only ~3 trees, because extra trees don't improve validation loss. With that few trees, many stocks get identical probabilities.
- **Testing the test.** An earlier version shuffled labels within each date but reported a pooled AUC. That mismatch once produced "p = 0.02" for an AUC *below* 0.5. Splitting it into two tests, each statistic matched to its own null, fixed it.
- **Why earlier numbers looked better (52–57%):** three likely contributors. (1) `validate.py` splits by *row* with a 5-row gap; with 5 symbols per day, that's about 1 day against a 5-day label, so boundary labels overlap. (2) Hyperparameters were chosen by Optuna on the same history, which adds selection bias. (3) A single short test window is noisy, with ±3pp swings between folds. `backtest.py` splits by date with a full horizon purge and reports results across 4 years.

This matches what the Efficient Market Hypothesis predicts for public technical indicators on the most heavily traded stocks in the world. A model claiming 80%+ accuracy here almost certainly has leakage.

**What could plausibly add signal:** a wider cross-section (100+ stocks) with a ranking target, excess-vs-sector returns instead of raw direction, alternative data (earnings surprises, news sentiment, options-implied skew), or less efficient universes (small caps).

| Ops metric | Value |
|---|---|
| Training time | 3–10 min (with tuning) |
| Prediction latency | < 200ms |

## 🔧 Configuration

Key settings in `src/utils/config.py` (can be overridden via `.env`):

```bash
SYMBOLS=AAPL,MSFT,GOOGL,AMZN,TSLA   # stocks to train on
LOOKBACK_DAYS=1825                    # 5 years of training data
PREDICTION_HORIZON=5                  # predict 5 trading days ahead
FEATURE_WINDOW=20                     # rolling window for indicators
```

## 🐳 Docker Deployment

```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## ☁️ Deploy to Hugging Face Spaces (Free)

### 1. Train locally and commit the model

```bash
python train.py
git add models/saved/latest_model.pkl models/best_params.json
git commit -m "Update trained model"
git push
```

### 2. Create a Space and sync from GitHub

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space), choose **Docker** SDK
2. Add your HF token as a GitHub secret (`HF_TOKEN`) in **repo Settings → Secrets → Actions**
3. Edit `.github/workflows/sync-to-hf.yml` — replace `YOUR_HF_USERNAME` with your username
4. In the Space **Settings → Variables**, add `DISABLE_TRAINING=true`

The pre-trained model is loaded on startup. To update, retrain locally, commit, and push.

## 🔍 Common Issues

**`403` on `http://localhost:5000` (MLflow UI)**
→ macOS AirPlay uses port 5000. Run `mlflow ui --port 5001` instead.

**All predictions show `low` confidence**
→ Expected when market has no clear directional signal. Only act on `medium`/`high` confidence predictions.

**Accuracy stuck at ~50% / Accuracy equals Precision**
→ Model may be predicting all UP. Re-tune with `python tune.py --metric val_precision`.

**Training fails with memory error**
→ Reduce `LOOKBACK_DAYS` in `.env` or use fewer symbols.

**API returns 503**
→ No model loaded. Run `python train.py` first or call `POST /train`.

## 📚 Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

## 📄 License

MIT License — free to use for portfolio and learning purposes.

---

**Built for ML Engineer Portfolio**
Demonstrates a production-grade ML pipeline: data ingestion, feature engineering with external signals, hyperparameter tuning, honest time-series evaluation, experiment tracking, and REST API serving.
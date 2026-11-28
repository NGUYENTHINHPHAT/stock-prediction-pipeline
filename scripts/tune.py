"""
Hyperparameter tuning with Optuna.
Searches XGBoost params on the purged validation split (never the test set),
saves them to models/best_params.json, then retrains and saves the final model.

Usage:
    python -m scripts.tune                         # 50 trials
    python -m scripts.tune --trials 100
    python -m scripts.tune --metric val_precision  # optimize precision instead of AUC
"""
import argparse
import json

import optuna
import optuna.logging

from src.utils.logger import log
from src.utils.config import settings
from src.models.xgboost_model import XGBoostModel
from src.models.trainer import ModelTrainer
from src.pipeline import PARAMS_FILE, build_dataset, train_and_save


def make_objective(df, metric: str):
    trainer = ModelTrainer(mlflow_tracking=False)
    train_df, val_df, _ = trainer.time_series_split(df, train_size=0.7, val_size=0.15)
    X_train, y_train = trainer.prepare_features_target(train_df)
    X_val, y_val = trainer.prepare_features_target(val_df)

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            # High n_estimators — early stopping decides the actual count
            "n_estimators": 1000,
        }

        model = XGBoostModel(params=params)
        model.train(X_train, y_train, X_val, y_val, early_stopping_rounds=30, verbose=False)

        metrics = trainer.calculate_metrics(
            y_val, model.predict(X_val), model.predict_proba(X_val)[:, 1], prefix="val_"
        )
        return metrics[metric]

    return objective


def main(symbols, lookback_days, n_trials, metric):
    log.info("=" * 70)
    log.info("HYPERPARAMETER TUNING")
    log.info(f"Trials: {n_trials}  |  Metric: {metric} (maximize)")
    log.info("=" * 70)

    df = build_dataset(symbols, lookback_days)

    # Suppress per-trial Optuna output — we print our own summary
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(df, metric), n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    log.info("\nBest trial:")
    log.info(f"  {metric}: {best.value:.4f}")
    log.info("  Params:")
    for k, v in best.params.items():
        log.info(f"    {k}: {v}")

    # Save best params so the train / validate / backtest scripts pick them up
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(best.params, indent=2))
    log.info(f"\nBest params saved -> {PARAMS_FILE}")

    log.info("\nRetraining final model with best params...")
    _, all_metrics = train_and_save(df, params=best.params, model_name="xgboost_tuned")

    log.info("\n" + "=" * 70)
    log.info("FINAL MODEL PERFORMANCE")
    log.info(f"{'Metric':<20} {'Train':>10} {'Val':>10} {'Test':>10}")
    log.info("-" * 55)
    for m in ["accuracy", "precision", "auc"]:
        log.info(
            f"{m.capitalize():<20}"
            f" {all_metrics.get(f'train_{m}', float('nan')):>10.4f}"
            f" {all_metrics.get(f'val_{m}', float('nan')):>10.4f}"
            f" {all_metrics.get(f'test_{m}', float('nan')):>10.4f}"
        )
    log.info("=" * 70)
    log.info("View all metrics in MLflow: mlflow ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument(
        "--metric",
        default="val_auc",
        choices=["val_auc", "val_accuracy", "val_precision"],
        help="Validation metric to maximize",
    )
    args = parser.parse_args()

    main(
        symbols=args.symbols or settings.symbol_list,
        lookback_days=args.lookback_days or settings.lookback_days,
        n_trials=args.trials,
        metric=args.metric,
    )
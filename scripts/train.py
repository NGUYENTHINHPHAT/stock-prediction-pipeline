"""
Train the production model and save it to models/saved/latest_model.pkl.

Uses a chronological 70/15/15 train/val/test split on dates, purged by the
prediction horizon. Runs are logged to MLflow (`mlflow ui` to browse).

Usage:
    python -m scripts.train
    python -m scripts.train --symbols AAPL MSFT --lookback-days 1095
    python -m scripts.train --params-file none        # XGBoost defaults
"""
import argparse

from src.pipeline import PARAMS_FILE, build_dataset, load_params, train_and_save
from src.utils.logger import log


def main(symbols=None, lookback_days=None, params_file=PARAMS_FILE):
    df = build_dataset(symbols, lookback_days)
    model, metrics = train_and_save(df, params=load_params(params_file))

    importance = model.get_feature_importance()
    if importance is not None:
        log.info("Top 10 features:\n" + importance.head(10).to_string(index=False))

    log.info("Held-out test set:")
    for k in ["accuracy", "auc", "precision", "recall", "f1"]:
        log.info(f"  {k:<10} {metrics[f'test_{k}']:.4f}")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--lookback-days", type=int)
    parser.add_argument("--params-file", default=str(PARAMS_FILE),
                        help="JSON of tuned XGBoost params, or 'none' for defaults")
    args = parser.parse_args()
    main(args.symbols, args.lookback_days,
         None if args.params_file.lower() == "none" else args.params_file)

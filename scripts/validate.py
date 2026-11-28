"""
Walk-forward cross-validation: expanding window, split on dates, purged by the
prediction horizon. Reports accuracy / precision / AUC per fold and averaged,
which is a far more honest estimate than a single train/test split.

Usage:
    python -m scripts.validate
    python -m scripts.validate --folds 8
    python -m scripts.validate --params-file none     # XGBoost defaults
"""
import argparse

from src.models.trainer import ModelTrainer
from src.pipeline import PARAMS_FILE, build_dataset, load_params
from src.utils.config import settings
from src.utils.logger import log


def main(symbols=None, n_splits=5, params_file=PARAMS_FILE):
    df = build_dataset(symbols)
    fold_metrics, avg = ModelTrainer(mlflow_tracking=False).walk_forward_validate(
        df=df,
        model_params=load_params(params_file),
        n_splits=n_splits,
        gap=settings.prediction_horizon,
    )

    log.info(f"{'Fold':<6} {'Train period':<28} {'Test period':<28}")
    for m in fold_metrics:
        log.info(f"{m['fold']:<6} {m['train_period']:<28} {m['test_period']:<28}")
    for k in ["accuracy", "precision", "auc"]:
        log.info(f"  Avg {k:<10} {avg[f'avg_{k}']:.4f} ± {avg[f'std_{k}']:.4f}")
    return fold_metrics, avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--params-file", default=str(PARAMS_FILE),
                        help="JSON of tuned XGBoost params, or 'none' for defaults")
    args = parser.parse_args()
    main(args.symbols, args.folds,
         None if args.params_file.lower() == "none" else args.params_file)

import logging
import os

import joblib
import numpy as np

from house_price.config import ARTIFACTS_DIR
from house_price.data.load import get_dataset
from house_price.evaluate import compute_regression_metrics
from house_price.logging_config import setup_logging
from house_price.models.train import train_and_evaluate_all
from house_price.models.tuning import tune_xgboost

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    df = get_dataset()

    results, data = train_and_evaluate_all(df)

    print("\nModel comparison (5-fold CV RMSE on log target, test metrics in dollars):")
    print(f"{'Model':<16} {'CV RMSE(log)':<16} {'Test MAE':<12} {'Test RMSE':<12} {'Test R2':<8}")
    for r in sorted(results, key=lambda r: r.test_metrics.rmse):
        print(
            f"{r.name:<16} {r.cv_rmse_log_mean:.4f}+-{r.cv_rmse_log_std:.4f}   "
            f"${r.test_metrics.mae:<10,.0f} ${r.test_metrics.rmse:<10,.0f} {r.test_metrics.r2:<8.3f}"
        )

    n_trials = int(os.environ.get("OPTUNA_TRIALS", "30"))
    print(f"\nTuning XGBoost with Optuna ({n_trials} trials, Bayesian optimization)...")
    y_train_log = np.log1p(data.y_train)
    study = tune_xgboost(data.X_train, y_train_log, n_trials=n_trials)
    print(f"Best params: {study.best_params}")
    print(f"Best CV RMSE(log): {study.best_value:.4f}")

    from xgboost import XGBRegressor

    tuned_model = XGBRegressor(random_state=42, n_jobs=-1, **study.best_params)
    tuned_model.fit(data.X_train, y_train_log)
    tuned_preds = np.expm1(tuned_model.predict(data.X_test))
    tuned_metrics = compute_regression_metrics(data.y_test, tuned_preds)
    print(
        f"\nTuned XGBoost test metrics: MAE=${tuned_metrics.mae:,.0f} "
        f"RMSE=${tuned_metrics.rmse:,.0f} R2={tuned_metrics.r2:.3f}"
    )

    baseline_xgb = next(r for r in results if r.name == "XGBoost")
    print(
        f"Baseline XGBoost test metrics: MAE=${baseline_xgb.test_metrics.mae:,.0f} "
        f"RMSE=${baseline_xgb.test_metrics.rmse:,.0f} R2={baseline_xgb.test_metrics.r2:.3f}"
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(data.pipeline, ARTIFACTS_DIR / "feature_pipeline.joblib")
    joblib.dump(tuned_model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(data.feature_columns, ARTIFACTS_DIR / "feature_columns.joblib")
    print(f"\nSaved pipeline, tuned model, and feature columns to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()

import logging

import joblib
import numpy as np

from house_price.config import ARTIFACTS_DIR
from house_price.data.load import get_dataset
from house_price.explain import (
    compute_shap_values,
    get_feature_names,
    plot_shap_importance_bar,
    plot_shap_summary,
    plot_shap_waterfall,
    top_features_by_mean_abs_shap,
)
from house_price.logging_config import setup_logging
from house_price.models.train import get_model_candidates, prepare_train_test_data

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    df = get_dataset()
    data = prepare_train_test_data(df)

    model_path = ARTIFACTS_DIR / "model.joblib"
    if model_path.exists():
        logger.info("Loading trained model from %s", model_path)
        model = joblib.load(model_path)
    else:
        logger.info("No saved model found, training a baseline XGBoost model")
        model = get_model_candidates()["XGBoost"]
        model.fit(data.X_train, np.log1p(data.y_train))

    feature_names = get_feature_names(data.pipeline)
    shap_values = compute_shap_values(model, data.X_test, feature_names)

    plot_shap_summary(shap_values)
    plot_shap_importance_bar(shap_values)
    plot_shap_waterfall(shap_values, index=0)

    print("\nTop 10 features by mean |SHAP value| (impact on log1p(SalePrice)):")
    for name, value in top_features_by_mean_abs_shap(shap_values, top_n=10):
        print(f"  {name:<30} {value:.4f}")

    print("\nFigures saved to reports/figures/")


if __name__ == "__main__":
    main()

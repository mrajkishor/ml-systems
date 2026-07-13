import numpy as np
import pytest
from xgboost import XGBRegressor

from house_price.explain import (
    compute_shap_values,
    plot_shap_importance_bar,
    plot_shap_summary,
    plot_shap_waterfall,
    top_features_by_mean_abs_shap,
)


@pytest.fixture
def fitted_model_and_data():
    rng = np.random.default_rng(0)
    n, p = 50, 4
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(scale=0.1, size=n)

    model = XGBRegressor(n_estimators=20, max_depth=2, random_state=0)
    model.fit(X, y)
    feature_names = [f"feature_{i}" for i in range(p)]
    return model, X, feature_names


def test_compute_shap_values_shape_matches_input(fitted_model_and_data):
    model, X, feature_names = fitted_model_and_data
    shap_values = compute_shap_values(model, X, feature_names)

    assert shap_values.values.shape == X.shape
    assert shap_values.feature_names == feature_names


def test_most_important_feature_is_detected(fitted_model_and_data):
    model, X, feature_names = fitted_model_and_data
    shap_values = compute_shap_values(model, X, feature_names)

    top = top_features_by_mean_abs_shap(shap_values, top_n=1)
    assert top[0][0] == "feature_0"


def test_plots_write_files(tmp_path, fitted_model_and_data):
    model, X, feature_names = fitted_model_and_data
    shap_values = compute_shap_values(model, X, feature_names)

    summary_path = plot_shap_summary(shap_values, out_dir=tmp_path)
    bar_path = plot_shap_importance_bar(shap_values, out_dir=tmp_path)
    waterfall_path = plot_shap_waterfall(shap_values, index=0, out_dir=tmp_path)

    assert summary_path.exists()
    assert bar_path.exists()
    assert waterfall_path.exists()

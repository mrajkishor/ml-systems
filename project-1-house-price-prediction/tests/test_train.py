import numpy as np
import pandas as pd

from house_price.models.train import (
    cross_validate_model,
    get_model_candidates,
    prepare_train_test_data,
    train_and_evaluate_all,
)


def make_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.normal(1500, 200, n)
    quality = rng.integers(1, 10, n)
    neighborhood = rng.choice(["A", "B", "C"], size=n)
    price = area * 80 + quality * 5000 + rng.normal(0, 5000, n)
    return pd.DataFrame(
        {
            "GrLivArea": area,
            "OverallQual": quality,
            "Neighborhood": neighborhood,
            "SalePrice": price,
        }
    )


def test_prepare_train_test_data_splits_and_transforms():
    df = make_df()
    data = prepare_train_test_data(df, test_size=0.25)

    assert data.X_train.shape[0] == len(data.y_train)
    assert data.X_test.shape[0] == len(data.y_test)
    assert data.X_train.shape[1] == data.X_test.shape[1]
    assert abs(len(data.y_test) / len(df) - 0.25) < 0.05


def test_get_model_candidates_has_expected_models():
    candidates = get_model_candidates()

    assert set(candidates) == {"LinearRegression", "Ridge", "Lasso", "RandomForest", "XGBoost"}


def test_cross_validate_model_returns_finite_scores():
    df = make_df()
    data = prepare_train_test_data(df)
    y_log = np.log1p(data.y_train)

    mean, std = cross_validate_model(get_model_candidates()["Ridge"], data.X_train, y_log, cv=3)

    assert np.isfinite(mean)
    assert np.isfinite(std)


def test_train_and_evaluate_all_runs_every_candidate():
    df = make_df()
    results, data = train_and_evaluate_all(df)

    names = {r.name for r in results}
    assert names == {"LinearRegression", "Ridge", "Lasso", "RandomForest", "XGBoost"}
    for r in results:
        assert r.test_metrics.r2 > 0.5

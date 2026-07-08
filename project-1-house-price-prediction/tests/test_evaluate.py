import numpy as np

from house_price.evaluate import compute_regression_metrics


def test_perfect_predictions_give_zero_error_and_r2_one():
    y_true = np.array([100.0, 200.0, 300.0])
    metrics = compute_regression_metrics(y_true, y_true)

    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.r2 == 1.0


def test_known_mae():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics.mae == 10.0
    assert metrics.rmse == 10.0

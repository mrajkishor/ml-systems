import numpy as np

from churn_prediction.features.imbalance import (
    apply_smote,
    compute_balanced_class_weight,
    find_best_threshold,
)


def test_compute_balanced_class_weight_favors_minority():
    y = np.array([0] * 90 + [1] * 10)
    weights = compute_balanced_class_weight(y)

    assert weights[1] > weights[0]


def test_apply_smote_balances_classes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = np.array([0] * 90 + [1] * 10)

    X_res, y_res = apply_smote(X, y)

    counts = np.bincount(y_res)
    assert counts[0] == counts[1]
    assert len(y_res) > len(y)


def test_find_best_threshold_on_separable_data():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])

    threshold, score = find_best_threshold(y_true, y_proba)

    assert 0.4 <= threshold <= 0.6
    assert score > 0.9

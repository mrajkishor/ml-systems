import numpy as np

from churn_prediction.evaluate import (
    compute_classification_metrics,
    plot_learning_curve,
    plot_pr_curve,
    plot_roc_curve,
)


def test_compute_classification_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0


def test_plot_roc_curve_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.concatenate([rng.uniform(0, 0.5, 50), rng.uniform(0.5, 1, 50)])

    path = plot_roc_curve(y_true, y_proba, label="test", out_dir=tmp_path)

    assert path.exists()


def test_plot_pr_curve_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.concatenate([rng.uniform(0, 0.5, 50), rng.uniform(0.5, 1, 50)])

    path = plot_pr_curve(y_true, y_proba, label="test", out_dir=tmp_path)

    assert path.exists()


def test_plot_learning_curve_writes_file(tmp_path):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] + rng.normal(scale=0.1, size=100) > 0).astype(int)

    path = plot_learning_curve(LogisticRegression(), X, y, cv=3, out_dir=tmp_path)

    assert path.exists()

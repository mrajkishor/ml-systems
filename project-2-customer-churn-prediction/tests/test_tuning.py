import numpy as np

from churn_prediction.models.tuning import tune_lightgbm


def test_tune_lightgbm_returns_search_with_best_params():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=150) > 0).astype(int)

    search = tune_lightgbm(X, y, n_iter=3, cv=2)

    assert search.best_params_
    assert np.isfinite(search.best_score_)

import numpy as np
import pandas as pd

from house_price.models.train import prepare_train_test_data
from house_price.models.tuning import tune_xgboost


def make_df(n=80, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.normal(1500, 200, n)
    quality = rng.integers(1, 10, n)
    price = area * 80 + quality * 5000 + rng.normal(0, 5000, n)
    return pd.DataFrame({"GrLivArea": area, "OverallQual": quality, "SalePrice": price})


def test_tune_xgboost_returns_study_with_best_params():
    df = make_df()
    data = prepare_train_test_data(df)
    y_log = np.log1p(data.y_train)

    study = tune_xgboost(data.X_train, y_log, n_trials=3, cv=2)

    assert study.best_params
    assert np.isfinite(study.best_value)

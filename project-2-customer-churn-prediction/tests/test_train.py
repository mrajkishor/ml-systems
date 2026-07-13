import numpy as np
import pandas as pd

from churn_prediction.models.train import (
    cross_validate_model,
    get_model_candidates,
    prepare_train_test_data,
    train_and_evaluate_all,
)


def make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    contract = np.where(tenure < 12, "Month-to-month", rng.choice(["One year", "Two year"], size=n))
    monthly = rng.normal(65, 20, n)
    churn_prob = 1 / (1 + np.exp(-(-tenure / 15 + (contract == "Month-to-month") * 2)))
    churn = rng.binomial(1, churn_prob)
    return pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Contract": contract,
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": (monthly * np.maximum(tenure, 1)).astype(str),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )


def test_prepare_train_test_data_stratifies_split():
    df = make_df()
    data = prepare_train_test_data(df, test_size=0.25)

    assert data.X_train.shape[0] == len(data.y_train)
    assert data.X_test.shape[0] == len(data.y_test)
    train_rate = data.y_train.mean()
    test_rate = data.y_test.mean()
    assert abs(train_rate - test_rate) < 0.15


def test_get_model_candidates_has_expected_models():
    candidates = get_model_candidates()

    assert set(candidates) == {"LogisticRegression", "RandomForest", "LightGBM", "CatBoost"}


def test_cross_validate_model_returns_finite_scores():
    df = make_df()
    data = prepare_train_test_data(df)

    mean, std = cross_validate_model(get_model_candidates()["LogisticRegression"], data.X_train, data.y_train, cv=3)

    assert np.isfinite(mean)
    assert np.isfinite(std)


def test_train_and_evaluate_all_runs_every_candidate():
    df = make_df()
    results, data = train_and_evaluate_all(df)

    names = {r.name for r in results}
    assert names == {"LogisticRegression", "RandomForest", "LightGBM", "CatBoost"}
    for r in results:
        assert r.test_metrics.roc_auc > 0.5

import numpy as np
import pandas as pd

from churn_prediction.features.engineering import build_features, encode_binary_and_ordinal


def make_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.normal(65, 20, n),
            "TotalCharges": (rng.normal(65, 20, n) * rng.integers(1, 72, n)).astype(str),
            "Churn": rng.choice(["Yes", "No"], size=n, p=[0.3, 0.7]),
        }
    )


def test_encode_binary_and_ordinal_maps_known_values():
    df = pd.DataFrame({"Partner": ["Yes", "No"], "Contract": ["Month-to-month", "Two year"]})
    result = encode_binary_and_ordinal(df)

    assert result["Partner"].tolist() == [1, 0]
    assert result["Contract"].tolist() == [0, 2]


def test_build_features_returns_binary_target_and_transformed_array():
    df = make_df()
    X, y, pipeline = build_features(df)

    assert X.shape[0] == len(df)
    assert set(y.unique()) <= {0, 1}
    assert not np.isnan(X).any()


def test_build_features_pipeline_is_reusable_on_new_data():
    df = make_df()
    X_train, _, pipeline = build_features(df)

    new_df = make_df(n=5, seed=1).drop(columns=["Churn"])
    X_new = pipeline.transform(new_df)

    assert X_new.shape[1] == X_train.shape[1]

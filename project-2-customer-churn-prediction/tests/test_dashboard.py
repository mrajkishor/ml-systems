import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from churn_prediction.dashboard import explain_one, get_feature_names, predict_one
from churn_prediction.features.engineering import build_feature_pipeline


def make_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.normal(65, 20, n),
        }
    )


def fitted_pipeline_and_model(df, seed=0):
    pipeline = build_feature_pipeline(df)
    X = pipeline.fit_transform(df)
    y = np.random.default_rng(seed).integers(0, 2, len(df))
    model = LGBMClassifier(n_estimators=10, max_depth=2, verbosity=-1)
    model.fit(X, y)
    return pipeline, model


def test_predict_one_returns_probability_and_prediction():
    df = make_df()
    pipeline, model = fitted_pipeline_and_model(df)

    result = predict_one(pipeline, model, threshold=0.5, input_dict=df.iloc[0].to_dict())

    assert 0 <= result["probability"] <= 1
    assert result["prediction"] in {"Churn", "No churn"}


def test_get_feature_names_matches_transformed_width():
    df = make_df()
    pipeline = build_feature_pipeline(df)
    X = pipeline.fit_transform(df)

    names = get_feature_names(pipeline)

    assert len(names) == X.shape[1]


def test_explain_one_returns_top_contributions():
    df = make_df()
    pipeline, model = fitted_pipeline_and_model(df)
    feature_names = get_feature_names(pipeline)
    X = pipeline.transform(df.iloc[[0]])

    contributions = explain_one(model, X, feature_names)

    assert len(contributions) <= 10
    assert "contribution" in contributions.columns

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from xgboost import XGBRegressor

from house_price.api import main as api_main
from house_price.features.engineering import build_feature_pipeline


@pytest.fixture
def client(monkeypatch):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "GrLivArea": rng.normal(1500, 200, 40),
            "OverallQual": rng.integers(1, 10, 40),
            "Neighborhood": rng.choice(["A", "B"], size=40),
        }
    )
    y_log = np.log1p(rng.normal(200000, 20000, 40))

    pipeline = build_feature_pipeline(df)
    X = pipeline.fit_transform(df)
    model = XGBRegressor(n_estimators=10, max_depth=2, random_state=0)
    model.fit(X, y_log)

    monkeypatch.setattr(api_main, "_load_artifacts", lambda: (pipeline, model, list(df.columns)))
    return TestClient(api_main.app)


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_positive_price(client):
    response = client.post(
        "/predict", json={"GrLivArea": 1600, "OverallQual": 7, "Neighborhood": "A"}
    )

    assert response.status_code == 200
    assert response.json()["predicted_sale_price"] > 0


def test_predict_handles_partial_input(client):
    response = client.post("/predict", json={"OverallQual": 8})

    assert response.status_code == 200
    assert response.json()["predicted_sale_price"] > 0


def test_predict_without_artifacts_returns_503(monkeypatch):
    def _raise():
        raise FileNotFoundError("Model artifacts not found.")

    monkeypatch.setattr(api_main, "_load_artifacts", _raise)
    client = TestClient(api_main.app)

    response = client.post("/predict", json={"OverallQual": 8})

    assert response.status_code == 503

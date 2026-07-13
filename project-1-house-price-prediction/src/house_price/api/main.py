import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from house_price.config import ARTIFACTS_DIR
from house_price.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="House Price Prediction API", version="0.1.0")


class PredictionRequest(BaseModel):
    """Raw Ames Housing feature values, e.g. `{"OverallQual": 7, "GrLivArea": 1800}`.

    Any subset of the training columns is accepted -- fields you omit are imputed the
    same way missing values were handled during training (median/mode/domain defaults).
    """

    model_config = ConfigDict(extra="allow")


class PredictionResponse(BaseModel):
    predicted_sale_price: float


@lru_cache(maxsize=1)
def _load_artifacts():
    pipeline_path = ARTIFACTS_DIR / "feature_pipeline.joblib"
    model_path = ARTIFACTS_DIR / "model.joblib"
    columns_path = ARTIFACTS_DIR / "feature_columns.joblib"

    if not (pipeline_path.exists() and model_path.exists() and columns_path.exists()):
        raise FileNotFoundError(
            "Model artifacts not found. Run `python scripts/run_training.py` first."
        )

    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)
    feature_columns = joblib.load(columns_path)
    return pipeline, model, feature_columns


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        pipeline, model, feature_columns = _load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame([payload.model_dump()]).reindex(columns=feature_columns)
    X = pipeline.transform(row)
    prediction = float(np.expm1(model.predict(X))[0])
    return PredictionResponse(predicted_sale_price=prediction)

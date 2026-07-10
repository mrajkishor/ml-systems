# Project 1: House Price Prediction — Full ML Pipeline

End-to-end regression pipeline on the Kaggle Ames Housing dataset: EDA, feature engineering,
model comparison (Linear/Ridge/Lasso/Random Forest/XGBoost), Optuna tuning, SHAP explainability,
and a FastAPI serving layer.

See the [Project 1 checklist](../Checklist/project_1.md) for the full task list.

## Structure

```
src/house_price/
  data/       # loading raw data
  features/   # feature engineering
  models/     # training & inference
  api/        # FastAPI app
tests/        # pytest suite
data/         # raw/processed data (gitignored contents)
notebooks/    # EDA notebooks
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest
```

## Run API

```bash
python scripts/run_training.py   # trains models and saves artifacts/ (pipeline, model, feature columns)
uvicorn house_price.api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"OverallQual": 7, "GrLivArea": 1800, "Neighborhood": "NAmes"}'
```

Any subset of the training columns is accepted; omitted fields are imputed the same way
missing values were handled during training.

## Docker

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

The image trains a model at build time (`OPTUNA_TRIALS=15`, kept low for build speed) so the
container is self-contained — no external artifact store needed.

## Deploy (Railway / Render)

This project lives in a subdirectory of a monorepo, so when connecting the repo you must set
the service's **root directory** to `project-1-house-price-prediction` — otherwise the platform
won't find the `Dockerfile`.

**Render:**
1. New → Web Service → connect the `ml-systems` GitHub repo
2. Root Directory: `project-1-house-price-prediction`
3. Environment: Docker (auto-detected from the `Dockerfile`)
4. Instance type: Free
5. Deploy — Render builds the image (runs training) and exposes the service on its assigned URL

**Railway:**
1. New Project → Deploy from GitHub repo → select `ml-systems`
2. Settings → Root Directory: `project-1-house-price-prediction`
3. Railway detects the `Dockerfile` automatically and builds/deploys
4. Add a public domain under Settings → Networking

Both require your own account connected to GitHub — deployment itself isn't something that can
be scripted from here.

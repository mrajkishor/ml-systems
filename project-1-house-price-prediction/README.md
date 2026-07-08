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
uvicorn house_price.api.main:app --reload
```

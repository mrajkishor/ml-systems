# Project 2: Customer Churn Prediction with Imbalanced Data

Binary classification on a telecom churn dataset, with a focus on handling extreme class
imbalance (SMOTE, class weights, threshold tuning), model comparison (Logistic Regression,
Random Forest, LightGBM, CatBoost), and business-framed cost-sensitive evaluation.

See the [Project 2 checklist](../Checklist/project_2.md) for the full task list.

## Structure

```
src/churn_prediction/
  data/       # loading raw data
  features/   # feature engineering, imbalance handling
  models/     # training & inference
  api/        # (reserved; this project serves via Streamlit, not FastAPI)
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

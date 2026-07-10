Found it in [projects.md](projects.md) — **Project 1: House Price Prediction — Full ML Pipeline** (Tier 1, ⭐☆☆☆☆). Here's the checklist:

## Project 1 Checklist: House Price Prediction — Full ML Pipeline

**Setup**
- [x] Repo structure: clean modules/packages, not a single notebook
- [x] Git initialized, `.gitignore`, meaningful commit history
- [x] `pytest` test suite scaffolded
- [x] Logging configured

**Data & EDA**
- [x] Load Kaggle Ames Housing dataset
- [x] Exploratory Data Analysis with Matplotlib/Plotly
- [x] Statistical hypothesis testing / confidence intervals in EDA (§1.3)

**Feature Engineering**
- [x] Missing value imputation
- [x] Categorical encoding
- [x] Feature scaling
- [x] Outlier treatment

**Modeling**
- [x] Linear Regression (baseline)
- [x] Ridge Regression
- [x] Lasso Regression
- [x] Random Forest
- [x] XGBoost
- [x] Cross-validation + bias-variance analysis
- [x] Metrics: MAE, RMSE, R²
- [x] Hyperparameter tuning with Optuna (Bayesian optimization)

**Explainability**
- [x] SHAP values for model interpretability

**Serving & Deployment**
- [x] FastAPI REST endpoint for predictions
- [x] Dockerize the app
- [ ] Deploy to Railway or Render (free tier) — needs your own account; steps documented in project README

**Polish (from "What Makes a Project Stand Out")**
- [x] README with problem statement, architecture diagram, results
- [ ] Live demo link — blocked on Railway/Render deployment (needs your account)
- [x] Before/after metrics (baseline vs. tuned model)
- [x] Write-up explaining decisions/tradeoffs (see `docs/WRITEUP.md`)
- [ ] 2-minute demo video — needs you to record it

**Key skills this proves:** EDA, feature engineering, ensemble models, model explainability, basic deployment.
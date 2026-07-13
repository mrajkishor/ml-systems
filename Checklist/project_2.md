## Project 2 Checklist: Customer Churn Prediction with Imbalanced Data

**Setup**
- [x] Repo structure: clean modules/packages, not a single notebook
- [x] Git initialized, `.gitignore`, meaningful commit history
- [x] `pytest` test suite scaffolded
- [x] Logging configured

**Data & EDA**
- [x] Load telecom/bank churn dataset
- [x] Exploratory Data Analysis with Pandas/Matplotlib
- [x] Class imbalance analysis (churn rate, class distribution)

**Feature Engineering**
- [x] Encoding, scaling, missing value handling
- [x] Handle class imbalance: SMOTE, class weights
- [x] Threshold tuning

**Modeling**
- [x] Logistic Regression (baseline)
- [x] Random Forest
- [x] LightGBM
- [x] CatBoost
- [x] Stratified k-fold cross-validation
- [x] Precision-Recall curves, AUC-ROC analysis
- [x] Hyperparameter tuning (grid/random search), learning curves

**Business Interpretation**
- [x] Cost-sensitive evaluation (business framing of false positives/negatives)

**Serving & Deployment**
- [x] Streamlit dashboard showing predictions + explanations

**Polish (from "What Makes a Project Stand Out")**
- [x] README with problem statement, architecture diagram, results
- [ ] Live demo link — blocked on Streamlit Community Cloud/Render deployment (needs your account)
- [x] Before/after metrics (tuning impact, class_weight vs SMOTE, threshold choice)
- [x] Write-up explaining decisions/tradeoffs (see `docs/WRITEUP.md`)
- [ ] 2-minute demo video — needs you to record it

**Key skills this proves:** Imbalanced learning, business framing, model evaluation depth.
